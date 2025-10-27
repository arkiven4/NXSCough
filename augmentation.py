import random
import torch
import torchaudio
import os
import logging

#region logging
# ---------------------------------------------------------------------
# Logging

# Set logging config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger_formatter = logging.Formatter(
    fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt = '%y-%m-%d %H:%M:%S',
    )

# Set a logging stream handler
logger_stream_handler = logging.StreamHandler()
logger_stream_handler.setLevel(logging.INFO)
logger_stream_handler.setFormatter(logger_formatter)

# Add handlers
logger.addHandler(logger_stream_handler)
# ---------------------------------------------------------------------
#endregion

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
        self.PITCH_SHIFTS = [-2, -1, 1, 2]         
        self.SPEEDS = ["0.9", "1.1"] # If 1 is an option, no augmentation is done!
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
    

    def apply_speed_perturbation(self, audio, sample_rate):
            
        speed = random.choice(self.SPEEDS)

        augmented_audio_waveform, augmented_sample_rate = torchaudio.sox_effects.apply_effects_tensor(
            audio, sample_rate, [["speed", speed]]
        )

        # Speed perturbation changes only the sampling rate, so we need to resample to the original sample rate
        resampled_waveform = torchaudio.functional.resample(
            waveform = augmented_audio_waveform,
            orig_freq = augmented_sample_rate, 
            new_freq = sample_rate, 
        )
        
        # We return the resampled waveform, the sample rate remains the original
        return resampled_waveform
    
    
    def apply_reverb(self, audio, sample_rate):
            
        if self.rirs_directory is not None:
            path = os.path.join(self.rirs_directory, random.choice(self.rirs_list).strip())
        else:
            path = random.choice(self.rirs_list).strip()
        logger.debug(f"path: {path}")
        try: 
            rir_wav, rir_sample_rate = torchaudio.load(path)
        except RuntimeError as e:
            self.number_of_misses += 1
            #logger.info(f"The number of misses that we have is {self.number_of_misses}")
            #logger.error(f"Error loading RIR from path: {path}. Error: {str(e)}")
            return audio

        logger.debug(f"first load ok")
        if rir_sample_rate != sample_rate:
            rir_wav = torchaudio.functional.resample(
                waveform = rir_wav,
                orig_freq = rir_sample_rate, 
                new_freq = sample_rate, 
            )
            rir_sample_rate = sample_rate
        logger.debug(f"resampling ok")

        # TODO first loading the audio and then cropping is unefficient
        # Clean up the RIR,  extract the main impulse, normalize the signal power
        normalized_rir = rir_wav[:, int(rir_sample_rate * 0.01) : int(rir_sample_rate * 1.3)]
        rir_norm = torch.norm(normalized_rir, p=2)
        if rir_norm > 0:
            normalized_rir = normalized_rir / rir_norm
        else:
            logger.warning("RIR norm is zero, skipping reverb")
            return audio
        
        logger.debug(f"fftconvolve on going...")
        logger.debug(f"The audio shape is: {audio.shape}")
        logger.debug(f"The normalized_rir shape:{normalized_rir.shape}")

        # there is an speciefic file that is causing trouble bc it is of shape [8, 56889]
        normalized_rir = torch.mean(normalized_rir, dim=0)
        normalized_rir = normalized_rir.view(1, -1)

        augmented_waveform = torch.mean(torchaudio.functional.fftconvolve(audio, normalized_rir), dim=0).unsqueeze(0)
        logger.debug(f"fftconvolve ok")

        return augmented_waveform

    def apply_pitch_shift(self, audio, sample_rate):
            """
            Apply pitch shift augmentation to the audio.
            """
            pitch_shift_semitones = random.choice(self.PITCH_SHIFTS)
            
            augmented_audio_waveform, augmented_sample_rate = torchaudio.sox_effects.apply_effects_tensor(
                audio, sample_rate, [["pitch", str(pitch_shift_semitones)]]
            )
            
            # Pitch shift typically doesn't change sample rate, but let's be safe
            if augmented_sample_rate != sample_rate:
                resampled_waveform = torchaudio.functional.resample(
                    waveform = augmented_audio_waveform,
                    orig_freq = augmented_sample_rate, 
                    new_freq = sample_rate, 
                )
                return resampled_waveform
            
            return augmented_audio_waveform
    
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
        logger.debug(f"path: {path}")
        noise, noise_sample_rate = torchaudio.load(path)
        logger.debug(f"first load ok")
        if noise_sample_rate != sample_rate:
            noise = torchaudio.functional.resample(
                waveform = noise,
                orig_freq = noise_sample_rate, 
                new_freq = sample_rate, 
            )
            noise_sample_rate = sample_rate
        logger.debug(f"resampling ok")

        # TODO first loading the audio and then cropping is unefficient
        cropped_noise = self.crop_noise(
            noise, 
            noise_sample_rate, 
            min(self.window_size_secs, int(audio.size()[1] / sample_rate)),
        )

        padded_cropped_noise = self.pad_noise(cropped_noise, audio)
        
        audio_power = torch.mean(audio ** 2)
        noise_power = torch.mean(padded_cropped_noise ** 2)
        if audio_power == 0 or noise_power == 0:
            logger.warning("Zero power detected in audio or noise, skipping noise addition")
            return audio
        
        audio_SNR = torch.tensor(
            self.sample_random_SNR(background_audio_type)
        ).unsqueeze(0)

        noisy_audio = torchaudio.functional.add_noise(audio, padded_cropped_noise, audio_SNR)

        return noisy_audio
        
    
    def augment(self, audio, sample_rate):
        """
        Returns a waveform with some random augmentation effect applied in it.

        The output shape is [1, len(waveform)]
        """
        if torch.isnan(audio).any():
            logger.warning("Input audio contains NaN values")
            return audio
        
        effect = random.choice(self.augmentation_effects)

        logger.debug(f"Data augmentation {effect} is going to be applied...")
        
        # getattr(self, effect) is equivalent to apply self.effect(audio, sample_rate)
        augmented_waveform = getattr(self, effect)(audio, sample_rate)
        if torch.isnan(augmented_waveform).any():
            logger.warning(f"Augmentation {effect} produced NaN values, returning original audio")
            return audio
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

        

        

        

        

        

        

        

        

        