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
        normalized_rir = normalized_rir / torch.norm(normalized_rir, p=2)
        
        logger.debug(f"fftconvolve on going...")
        logger.debug(f"The audio shape is: {audio.shape}")
        logger.debug(f"The normalized_rir shape:{normalized_rir.shape}")

        # there is an speciefic file that is causing trouble bc it is of shape [8, 56889]
        normalized_rir = torch.mean(normalized_rir, dim=0)
        normalized_rir = normalized_rir.view(1, -1)

        augmented_waveform = torch.mean(torchaudio.functional.fftconvolve(audio, normalized_rir), dim=0).unsqueeze(0)
        logger.debug(f"fftconvolve ok")

        return augmented_waveform
            
    
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
        
        effect = random.choice(self.augmentation_effects)

        logger.debug(f"Data augmentation {effect} is going to be applied...")
        
        # getattr(self, effect) is equivalent to apply self.effect(audio, sample_rate)

        augmented_waveform = getattr(self, effect)(audio, sample_rate)
        
        return augmented_waveform

    
    def __call__(self, audio, sample_rate):
        
        return self.augment(audio, sample_rate)

        
    def __len__(self):

        return len(self.augmentation_list)

        

        

        

        

        

        

        

        

        

        