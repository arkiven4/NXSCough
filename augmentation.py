# https://github.com/marccasals98/BSC-UPC_EmoSPeech/blob/main/src/augmentation.py

import random
import torch
import torchaudio
import os
import logging
import librosa

import warnings

# Suppress torchaudio deprecation warnings
warnings.filterwarnings("ignore", message=".*torchaudio.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*StreamingMediaDecoder.*", category=UserWarning)

# region logging
# ---------------------------------------------------------------------
# Logging

# Set logging config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger_formatter = logging.Formatter(
    fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%y-%m-%d %H:%M:%S',
)

# Set a logging stream handler
logger_stream_handler = logging.StreamHandler()
logger_stream_handler.setLevel(logging.INFO)
logger_stream_handler.setFormatter(logger_formatter)

# Add handlers
logger.addHandler(logger_stream_handler)
# ---------------------------------------------------------------------
# endregion

AUGMENTATION_ORDER = {
    "apply_speed_perturbation": 0,
    "apply_pitch_shift": 1,
    "apply_reverb": 2,
    "add_background_noise": 3,
    "apply_random_gain": 4,
}

class DataAugmentator:
    def __init__(
        self,
        augmentation_noises_directory,
        augmentation_noises_labels_path,
        augmentation_rirs_directory,
        augmentation_rirs_labels_path,
        augmentation_window_size_secs,
        augmentation_probability,
    ):
        # Background noises directory
        self.augmentation_directory = augmentation_noises_directory
        self.rirs_directory = augmentation_rirs_directory  # RIRs directory
        self.window_size_secs = augmentation_window_size_secs
        self.augmentation_effects = ["apply_speed_perturbation", "apply_pitch_shift", "apply_reverb", "add_background_noise", "apply_random_gain"]
        self.augmentation_probability = augmentation_probability

        self.create_augmentation_list(augmentation_noises_labels_path)
        self.create_rir_list(augmentation_rirs_labels_path)

        # TODO move to settings
        # self.EFFECTS = ["apply_speed_perturbation", "apply_reverb", "add_background_noise"]
        # If 1 is an option, no augmentation is done!
        self.PITCH_SHIFTS = [-1, 1]
        self.SPEEDS = [0.9, 1.1]
        self.SNR_NOISE_RANGE = [15, 25]
        self.SNR_SPEECH_RANGE = [15, 25]
        self.SNR_MUSIC_RANGE = [15, 20]
        self.number_of_misses = 0

    def create_augmentation_list(self, augmentation_labels_path):
        with open(augmentation_labels_path) as handle:
            self.augmentation_list = handle.readlines()

    def create_rir_list(self, rirs_labels_path):
        with open(rirs_labels_path) as handle:
            self.rirs_list = handle.readlines()

    def load_audio_safe(self, path):
        """
        Load audio safely and return as torch tensor.
        """
        data, sample_rate = torchaudio.load(path)
        data = data.float()
        if data.size(0) > 1:
            data = data.mean(dim=0, keepdim=True)  # [1, N]
        else:
            data = data.view(1, -1)  # force [1, N]
        return data, sample_rate

    def symmetric_soft_clip(self, wav, eps=1e-8):
        """
        wav: Tensor with shape [1, T] or [B, 1, T]
        returns: same shape, symmetric, softly clipped
        """
        peak = wav.abs().max()
        wav = wav / (peak + eps)
        return torch.tanh(wav)

    def apply_speed_perturbation(self, audio, sample_rate):
        """Speed perturbation implemented with librosa.

        This replaces `torchaudio.sox_effects.apply_effects_tensor` to avoid
        depending on SoX-enabled torchaudio builds.
        """
        speed = float(random.choice(self.SPEEDS))
        original_length = audio.shape[-1]

        waveform = audio
        audio_np = waveform.squeeze(0).numpy().astype(np.float32, copy=False)
        augmented_np = librosa.effects.time_stretch(audio_np, rate=speed)
        augmented = torch.from_numpy(augmented_np).to(audio.device)
        if audio.is_floating_point():
            augmented = augmented.to(dtype=audio.dtype)
        else:
            augmented = augmented.to(dtype=torch.float32)

        augmented = augmented[:original_length]
        return augmented.unsqueeze(0)

    def apply_pitch_shift(self, audio, sample_rate):
        """Pitch shift implemented with librosa.

        Semitone-based pitch shifting without changing duration.
        """
        n_steps = float(random.choice(self.PITCH_SHIFTS))  # e.g. [-2, -1, 1, 2]

        waveform = audio
        audio_np = waveform .squeeze(0) .numpy() .astype(np.float32, copy=False)

        augmented_np = librosa.effects.pitch_shift(
            audio_np,
            sr=sample_rate,
            n_steps=n_steps
        )
        augmented = torch.from_numpy(augmented_np).to(audio.device)
        if audio.is_floating_point():
            augmented = augmented.to(dtype=audio.dtype)
        else:
            augmented = augmented.to(dtype=torch.float32)

        return augmented.unsqueeze(0)


    def apply_reverb(self, audio, sample_rate):
        # ------------------------------------------------------------
        # Guard rails
        # ------------------------------------------------------------
        if audio.numel() == 0:
            return audio

        rir_line = random.choice(self.rirs_list).strip()
        rir_relpath = rir_line.split("\t")[0].strip()
        path = (
            os.path.join(self.rirs_directory, rir_relpath)
            if self.rirs_directory is not None
            else rir_relpath
        )

        try:
            rir_wav, rir_sample_rate = self.load_audio_safe(path)
        except:
            return audio
        
        if rir_wav.numel() == 0 or rir_wav.abs().max() < 1e-6:
            return audio

        # ------------------------------------------------------------
        # Resample if needed
        # ------------------------------------------------------------
        if rir_sample_rate != sample_rate:
            rir_wav = torchaudio.functional.resample(
                rir_wav, rir_sample_rate, sample_rate
            )

        # ensure mono [1, N]
        if rir_wav.ndim == 2 and rir_wav.shape[0] > 1:
            rir_wav = rir_wav.mean(dim=0, keepdim=True)
        elif rir_wav.ndim == 1:
            rir_wav = rir_wav.unsqueeze(0)

        eps = 1e-8

        # ------------------------------------------------------------
        # RIR preprocessing (PHYSICALLY PLAUSIBLE)
        # ------------------------------------------------------------

        # remove DC
        rir_wav = rir_wav - rir_wav.mean(dim=-1, keepdim=True)

        # trim leading silence using energy rise (NOT max peak)
        energy = torch.cumsum(rir_wav ** 2, dim=-1)
        energy = energy / (energy[..., -1:] + eps)

        # first index reaching 1% cumulative energy
        start_idx = int((energy >= 0.01).nonzero(as_tuple=True)[-1][0].item())

        # keep up to 1.2 seconds max
        max_len = int(1.2 * sample_rate)
        end_idx = min(start_idx + max_len, rir_wav.shape[-1])
        rir_wav = rir_wav[..., start_idx:end_idx]

        # fade out tail smoothly
        fade_len = int(0.05 * sample_rate)
        if rir_wav.shape[-1] > fade_len:
            fade = torch.ones(rir_wav.shape[-1], device=rir_wav.device)
            fade[-fade_len:] = torch.linspace(
                1.0, 0.0, fade_len, device=rir_wav.device
            )
            rir_wav = rir_wav * fade

        # ------------------------------------------------------------
        # Energy normalization (CRITICAL)
        # ------------------------------------------------------------
        rir_energy = torch.sqrt(torch.sum(rir_wav ** 2) + eps)
        rir_wav = rir_wav / rir_energy

        # conservative wet gain
        rir_wav = rir_wav * 0.25

        # ------------------------------------------------------------
        # Convolution (NO PEAK ALIGNMENT)
        # ------------------------------------------------------------
        full = torchaudio.functional.fftconvolve(audio, rir_wav)

        # causal crop: keep original length
        augmented = full[..., : audio.shape[-1]]

        # ------------------------------------------------------------
        # Loudness safety (CLAMPED, NON-DESTRUCTIVE)
        # ------------------------------------------------------------
        in_rms = torch.sqrt(torch.mean(audio ** 2) + eps)
        out_rms = torch.sqrt(torch.mean(augmented ** 2) + eps)

        gain = in_rms / (out_rms + eps)
        gain = torch.clamp(gain, 0.7, 1.3)

        augmented = augmented * gain
        return augmented


    def get_SNR_bounds(self, background_audio_type):
        if background_audio_type == "noise":
            return self.SNR_NOISE_RANGE
        elif background_audio_type == "speech":
            return self.SNR_SPEECH_RANGE
        elif background_audio_type == "music":
            return self.SNR_MUSIC_RANGE
        else:
            return self.SNR_NOISE_RANGE

    def sample_random_SNR(self, background_audio_type):
        snr_bounds = self.get_SNR_bounds(background_audio_type)
        return random.uniform(snr_bounds[0], snr_bounds[1])

    def crop_noise(self, noise, noise_sample_rate, window_size_secs):
        noise_duration_samples = noise.size()[1]
        noise_duration_secs = noise_duration_samples / noise_sample_rate
        window_size_samples = int(window_size_secs * noise_sample_rate)

        if noise_duration_secs <= window_size_secs:
            cropped_noise = noise[:, :]
        else:
            start = random.randint(0, noise_duration_samples - window_size_samples)
            end = start + window_size_samples
            cropped_noise = noise[:, start:end]

        return cropped_noise

    def pad_noise(self, noise, audio):
        # noise, audio: [1, N]
        target_len = audio.shape[1]
        noise_len = noise.shape[1]
        if noise_len >= target_len:
            return noise[:, :target_len]

        # repeat noise to exceed target length
        repeat_factor = (target_len + noise_len - 1) // noise_len
        noise_repeated = noise.repeat(1, repeat_factor)
        return noise_repeated[:, :target_len]

    def add_background_noise(self, audio, sample_rate):
        background_audio_line = random.choice(self.augmentation_list).strip()
        background_audio_name = background_audio_line.split("\t")[0].strip()
        background_audio_type = background_audio_line.split("\t")[1].strip().lower()
        if self.augmentation_directory is not None:
            path = os.path.join(self.augmentation_directory, background_audio_name)
        else:
            path = background_audio_name
        
        try:
            noise, noise_sample_rate = self.load_audio_safe(path)
        except:
            return audio
        
        if noise.numel() == 0 or torch.max(torch.abs(noise)) < 1e-8:
            print(f"Warning: Silent or empty noise file {path}, returning original audio")
            return audio
        
        if noise_sample_rate != sample_rate:
            noise = torchaudio.functional.resample(waveform=noise, orig_freq=noise_sample_rate,
                new_freq=sample_rate)
            noise_sample_rate = sample_rate

        # TODO first loading the audio and then cropping is unefficient
        cropped_noise = self.crop_noise(
            noise,
            noise_sample_rate,
            min(self.window_size_secs, audio.shape[-1] / sample_rate),
        )
        padded_cropped_noise = self.pad_noise(cropped_noise, audio)
        audio_SNR = torch.tensor(self.sample_random_SNR(background_audio_type)).unsqueeze(0)
        noisy_audio = torchaudio.functional.add_noise(audio, padded_cropped_noise, audio_SNR)

        return noisy_audio

    def augment(self, audio, sample_rate, return_effects: bool = False):
        """
        Returns a waveform with some random augmentation effect applied in it.

        The Input shape is [1, len(waveform)]
        The output shape is [1, len(waveform)]
        """
        if len(self.augmentation_probability) != len(self.augmentation_effects):
            raise ValueError(f"Number of probabilities ({len(self.augmentation_probability)}) must match number of effects ({len(self.augmentation_effects)})")
        
        available_effects = self.augmentation_effects.copy()
        available_probabilities = self.augmentation_probability.copy()
        num_effects = random.randint(1, 2)

        selected_effects = random.choices(
            available_effects,
            weights=available_probabilities,
            k=num_effects
        )
        selected_effects = list(dict.fromkeys(selected_effects))
        selected_effects.sort(key=lambda x: AUGMENTATION_ORDER[x])

        augmented_waveform = audio
        applied_effects = []

        # Step 3: apply in meaningful order
        for effect in selected_effects:
            augmented_waveform = getattr(self, effect)(
                augmented_waveform, sample_rate
            )
            applied_effects.append(effect)

        augmented_waveform = self.symmetric_soft_clip(augmented_waveform)
        if return_effects:
            return augmented_waveform, applied_effects
        return augmented_waveform

    def __call__(self, audio, sample_rate, return_effects: bool = False):
        return self.augment(audio, sample_rate, return_effects=return_effects)

    def __len__(self):
        return len(self.augmentation_list)

#########################################
#
# RAWBOOST
#
########################################

import numpy as np
from scipy import signal
import copy

def randRange(x1, x2, integer):
    y = np.random.uniform(low=x1, high=x2, size=(1,))
    if integer:
        y = int(y)
    return y

def normWav(x,always):
    if always:
        x = x/np.amax(abs(x))
    elif np.amax(abs(x)) > 1:
            x = x/np.amax(abs(x))
    return x

def genNotchCoeffs(nBands,minF,maxF,minBW,maxBW,minCoeff,maxCoeff,minG,maxG,fs):
    b = 1
    for i in range(0, nBands):
        fc = randRange(minF,maxF,0);
        bw = randRange(minBW,maxBW,0);
        c = randRange(minCoeff,maxCoeff,1);
          
        if c/2 == int(c/2):
            c = c + 1
        f1 = fc - bw/2
        f2 = fc + bw/2
        if f1 <= 0:
            f1 = 1/1000
        if f2 >= fs/2:
            f2 =  fs/2-1/1000
        b = np.convolve(signal.firwin(c, [float(f1), float(f2)], window='hamming', fs=fs),b)

    G = randRange(minG,maxG,0); 
    _, h = signal.freqz(b, 1, fs=fs)    
    b = pow(10, G/20)*b/np.amax(abs(h))   
    return b


def filterFIR(x,b):
    N = b.shape[0] + 1
    xpad = np.pad(x, (0, N), 'constant')
    y = signal.lfilter(b, 1, xpad)
    y = y[int(N/2):int(y.shape[0]-N/2)]
    return y

# Linear and non-linear convolutive noise
def LnL_convolutive_noise(x,N_f,nBands,minF,maxF,minBW,maxBW,minCoeff,maxCoeff,minG,maxG,minBiasLinNonLin,maxBiasLinNonLin,fs):
    y = [0] * x.shape[0]
    for i in range(0, N_f):
        if i == 1:
            minG = minG-minBiasLinNonLin;
            maxG = maxG-maxBiasLinNonLin;
        b = genNotchCoeffs(nBands,minF,maxF,minBW,maxBW,minCoeff,maxCoeff,minG,maxG,fs)
        y = y + filterFIR(np.power(x, (i+1)),  b)     
    y = y - np.mean(y)
    y = normWav(y,0)
    return y


# Impulsive signal dependent noise
def ISD_additive_noise(x, P, g_sd):
    beta = randRange(0, P, 0)
    
    y = copy.deepcopy(x)
    x_len = x.shape[0]
    n = int(x_len*(beta/100))
    p = np.random.permutation(x_len)[:n]
    f_r= np.multiply(((2*np.random.rand(p.shape[0]))-1),((2*np.random.rand(p.shape[0]))-1))
    r = g_sd * x[p] * f_r
    y[p] = x[p] + r
    y = normWav(y,0)
    return y


# Stationary signal independent noise
def SSI_additive_noise(x,SNRmin,SNRmax,nBands,minF,maxF,minBW,maxBW,minCoeff,maxCoeff,minG,maxG,fs):
    noise = np.random.normal(0, 1, x.shape[0])
    b = genNotchCoeffs(nBands,minF,maxF,minBW,maxBW,minCoeff,maxCoeff,minG,maxG,fs)
    noise = filterFIR(noise, b)
    noise = normWav(noise,1)
    SNR = randRange(SNRmin, SNRmax, 0)
    noise = noise / np.linalg.norm(noise,2) * np.linalg.norm(x,2) / 10.0**(0.05 * SNR)
    x = x + noise
    return x