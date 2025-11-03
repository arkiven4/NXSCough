import random
import torch
import os
import logging
import numpy as np
from scipy import signal
import librosa
import soundfile as sf

class DataAugmentator:
    def __init__(
        self,
        augmentation_noises_directory,
        augmentation_noises_labels_path,
        augmentation_rirs_directory,
        augmentation_rirs_labels_path,
        augmentation_window_size_secs,
        augmentation_effects,
    ):
        
        self.augmentation_directory = augmentation_noises_directory # Background noises directory
        self.rirs_directory = augmentation_rirs_directory # RIRs directory
        self.window_size_secs = augmentation_window_size_secs
        self.augmentation_effects = augmentation_effects

        self.create_augmentation_list(augmentation_noises_labels_path)
        self.create_rir_list(augmentation_rirs_labels_path) 

        # TODO move to settings
        #self.EFFECTS = ["apply_speed_perturbation", "apply_reverb", "add_background_noise"]    
        self.PITCH_SHIFTS = [-3, -2, -1, 1, 2, 3]          
        self.SPEEDS = ["0.9", "1.1"] # If 1 is an option, no augmentation is done!
        self.SNR_NOISE_RANGE = [15, 35]
        self.SNR_SPEECH_RANGE = [15, 35]
        self.SNR_MUSIC_RANGE = [15, 35]
        self.GAIN_DB = [
            (-18.0, -6.0),  # strong attenuation
            (-12.0, -3.0),  # strong attenuation
            (-6.0, 6.0),    # mild variation
            (3.0, 12.0)     # strong amplification
        ]
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
        data, sample_rate = sf.read(path, dtype='float32')  # always float32
        # If stereo, convert to mono by averaging channels
        if len(data.shape) > 1:
            data = data.mean(axis=1)
        audio_tensor = torch.from_numpy(data)
        return audio_tensor, sample_rate

    def apply_speed_perturbation(self, audio, sample_rate):
        speed = float(random.choice(self.SPEEDS))
        audio_np = audio.squeeze().numpy() if torch.is_tensor(audio) else audio
        perturbed = librosa.effects.time_stretch(audio_np, rate=speed)
        return torch.tensor(perturbed, dtype=torch.float32).unsqueeze(0) if torch.is_tensor(audio) else perturbed
    
    
    def apply_reverb(self, audio, sample_rate):
        if self.rirs_directory is not None:
            path = os.path.join(self.rirs_directory, random.choice(self.rirs_list).strip())
        else:
            path = random.choice(self.rirs_list).strip()
        try: 
            rir_wav, rir_sample_rate = self.load_audio_safe(path)
        except RuntimeError as e:
            self.number_of_misses += 1
            #logger.info(f"The number of misses that we have is {self.number_of_misses}")
            #logger.error(f"Error loading RIR from path: {path}. Error: {str(e)}")
            return audio

        if rir_sample_rate != sample_rate:
            # Convert to numpy for librosa resampling
            if isinstance(rir_wav, torch.Tensor):
                rir_np = rir_wav.squeeze().cpu().numpy()
            else:
                rir_np = rir_wav
            resampled_rir = librosa.resample(rir_np, orig_sr=rir_sample_rate, target_sr=sample_rate)
            rir_wav = torch.from_numpy(resampled_rir).float().unsqueeze(0)
            rir_sample_rate = sample_rate

        # TODO first loading the audio and then cropping is unefficient
        # Clean up the RIR,  extract the main impulse, normalize the signal power
        normalized_rir = rir_wav[:, int(rir_sample_rate * 0.01) : int(rir_sample_rate * 1.3)]
        normalized_rir = normalized_rir / torch.norm(normalized_rir, p=2)
    

        # there is an speciefic file that is causing trouble bc it is of shape [8, 56889]
        normalized_rir = torch.mean(normalized_rir, dim=0)
        normalized_rir = normalized_rir.view(1, -1)

        # Convert to numpy for scipy fftconvolve
        audio_np = audio.squeeze().cpu().numpy() if isinstance(audio, torch.Tensor) else audio.squeeze()
        rir_np = normalized_rir.squeeze().cpu().numpy() if isinstance(normalized_rir, torch.Tensor) else normalized_rir.squeeze()
        
        # Apply convolution using scipy
        convolved = signal.fftconvolve(audio_np, rir_np, mode='full')
        
        # Convert back to torch tensor
        augmented_waveform = torch.from_numpy(convolved).float().unsqueeze(0)

        return augmented_waveform

    def apply_pitch_shift(self, audio, sample_rate):
        """
        Apply a random pitch shift to the input audio.

        Args:
            audio (np.ndarray or torch.Tensor): 1D audio array
            sample_rate (int): Sample rate of the audio

        Returns:
            np.ndarray: Pitch-shifted audio
        """
        if isinstance(audio, np.ndarray):
            audio_np = audio.astype(float)  # librosa requires float
        else:
            audio_np = audio.cpu().numpy().astype(float)

        # Choose a random pitch shift
        n_steps = random.choice(self.PITCH_SHIFTS)

        # Apply pitch shift
        shifted_audio = librosa.effects.pitch_shift(audio_np, sr=sample_rate, n_steps=n_steps)

        return torch.from_numpy(shifted_audio).float()
    
    def apply_random_gain(self, audio):
            """
            Apply random gain to waveform.

            Args:
                audio (torch.Tensor): waveform tensor of shape [1, N] or [C, N]
                gain_db_range (tuple): min and max gain in decibels

            Returns:
                torch.Tensor: waveform with random gain applied
            """
            gain_db_range = random.choice(self.GAIN_DB)
            gain_db = random.uniform(*gain_db_range)
            gain = 10 ** (gain_db / 20)
            augmented_audio = torch.clamp(audio * gain, -1.0, 1.0)
            return augmented_audio
    
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
        
        if noise_duration_secs <=  window_size_secs:
            cropped_noise = noise[:, :]
        else:
            start = random.randint(0, noise_duration_samples - window_size_samples)
            end = start + window_size_samples
            cropped_noise = noise[:, start : end]
        
        return cropped_noise
    

    def pad_noise(self, noise, audio):

        pad_left = max(0, audio.shape[1] - noise.shape[1])

        cropped_noise_padded = torch.nn.functional.pad(noise, (pad_left, 0), mode = "constant")

        return cropped_noise_padded
    
    
    def add_background_noise(self, audio, sample_rate):
            
        background_audio_line = random.choice(self.augmentation_list).strip()

        background_audio_name = background_audio_line.split("\t")[0].strip()
        background_audio_type = background_audio_line.split("\t")[1].strip().lower()

        if self.augmentation_directory is not None:
            path = os.path.join(self.augmentation_directory, background_audio_name)
        else:
            path = background_audio_name
        noise, noise_sample_rate = self.load_audio_safe(path)
        if noise_sample_rate != sample_rate:
            # Convert to numpy for librosa resampling
            if isinstance(noise, torch.Tensor):
                noise_np = noise.squeeze().cpu().numpy()
            else:
                noise_np = noise
            resampled_noise = librosa.resample(noise_np, orig_sr=noise_sample_rate, target_sr=sample_rate)
            noise = torch.from_numpy(resampled_noise).float().unsqueeze(0)
            noise_sample_rate = sample_rate

        # TODO first loading the audio and then cropping is unefficient
        cropped_noise = self.crop_noise(
            noise, 
            noise_sample_rate, 
            min(self.window_size_secs, int(audio.size()[1] / sample_rate)),
        )

        padded_cropped_noise = self.pad_noise(cropped_noise, audio)

        audio_SNR_db = self.sample_random_SNR(background_audio_type)
        
        # Manual implementation of add_noise functionality
        # Calculate signal and noise power
        signal_power = torch.mean(audio ** 2)
        noise_power = torch.mean(padded_cropped_noise ** 2)
        
        # Calculate scaling factor based on desired SNR
        snr_linear = 10 ** (audio_SNR_db / 10.0)
        scaling_factor = torch.sqrt(signal_power / (noise_power * snr_linear))
        
        # Add scaled noise to signal
        noisy_audio = audio + padded_cropped_noise * scaling_factor

        return noisy_audio
        
    
    def augment(self, audio, sample_rate):
        """
        Returns a waveform with some random augmentation effect applied in it.

        The output shape is [1, len(waveform)]
        """
        
        effect = random.choice(self.augmentation_effects)
        
        # getattr(self, effect) is equivalent to apply self.effect(audio, sample_rate)

        augmented_waveform = getattr(self, effect)(audio, sample_rate)
        
        return augmented_waveform

    def __call__(self, audio, sample_rate):
        return self.augment(audio, sample_rate)

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

        

        

        

        

        

        

        

        

        