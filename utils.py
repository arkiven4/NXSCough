import math, librosa
import random
import string

import torch
from torchaudio import transforms as T

def generate_random_code(length=3):
    chars = string.ascii_uppercase + string.digits
    return ''.join(random.choices(chars, k=length))

def cut_pad_sample_torchaudio(data, sample_rate, desired_length, pad_types='zero', right_pad_shift_sec=0.04):
    fade_samples_ratio = 6
    fade_samples = int(sample_rate / fade_samples_ratio)
    fade_out = T.Fade(fade_in_len=0, fade_out_len=fade_samples, fade_shape='linear')
    target_duration = int(desired_length * sample_rate)
    current_length = data.shape[-1]

    if data.shape[-1] > target_duration:
        data = data[..., :target_duration]
    else:
        if pad_types == 'zero':
            total_pad = target_duration - current_length
            left_shift_samples = int(right_pad_shift_sec * sample_rate)
            pad_left = max((total_pad // 2) - left_shift_samples, 0)
            pad_right = total_pad - pad_left
            data = torch.nn.functional.pad(data, (pad_left, pad_right), mode='constant', value=0.0)

        elif pad_types == 'repeat':
            ratio = math.ceil(target_duration / data.shape[-1])
            data = data.repeat(1, ratio)
            data = data[..., :target_duration]
            data = fade_out(data)
    
    return data

def load_audio_sample(file_path, sample_rate, wav_stats, desired_length, fade_samples_ratio=6, pad_types='zero'):
    data, sample_rate = librosa.load(file_path, sr=sample_rate)
    data = data / 32768.0
    data = (data - wav_stats[0]) / (wav_stats[1] + 0.000001)

    data = torch.from_numpy(data).unsqueeze(0)

    fade_samples = int(sample_rate / fade_samples_ratio)
    fade = T.Fade(fade_in_len=fade_samples, fade_out_len=fade_samples, fade_shape='linear')
    data = fade(data)
    data = cut_pad_sample_torchaudio(data, sample_rate, desired_length, pad_types=pad_types)
    return data