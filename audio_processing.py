import torch, random
import numpy as np
from scipy.signal import get_window
import librosa.util as librosa_util

def compute_gain(sound, fs, min_db=-80.0, mode='A_weighting'):
    if fs == 16000:
        n_fft = 2048
    elif fs == 44100:
        n_fft = 4096
    else:
        raise Exception('Invalid fs {}'.format(fs))
    stride = n_fft // 2

    gain = []
    for i in range(0, len(sound) - n_fft + 1, stride):
        if mode == 'RMSE':
            g = np.mean(sound[i: i + n_fft] ** 2)
        elif mode == 'A_weighting':
            spec = np.fft.rfft(np.hanning(n_fft + 1)[:-1] * sound[i: i + n_fft])
            power_spec = np.abs(spec) ** 2
            a_weighted_spec = power_spec * np.power(10, a_weight(fs, n_fft) / 10)
            g = np.sum(a_weighted_spec)
        else:
            raise Exception('Invalid mode {}'.format(mode))
        gain.append(g)

    gain = np.array(gain)
    gain = np.maximum(gain, np.power(10, min_db / 10))
    gain_db = 10 * np.log10(gain)

    return gain_db

def mix(sound1, sound2, r, fs):
    gain1 = np.max(compute_gain(sound1, fs))  # Decibel
    gain2 = np.max(compute_gain(sound2, fs))
    t = 1.0 / (1 + np.power(10, (gain1 - gain2) / 20.) * (1 - r) / r)
    sound = ((sound1 * t + sound2 * (1 - t)) / np.sqrt(t ** 2 + (1 - t) ** 2))

    return sound

def multi_mask_spectrogram(X, tau=30, nu=13, num_masks=2):
    """
    Apply multiple time and frequency masks to a spectrogram.

    Args:
        X (Tensor): Spectrogram of shape (F, T) or (B, F, T).
        tau (int): Time mask parameter (τ) – max number of time steps to mask.
        nu (int): Frequency mask parameter (ν) – max number of frequency bins to mask.
        num_masks (int): Number of time/frequency masks to apply (N).

    Returns:
        Tensor: Augmented spectrogram with shape same as input.
    """
    NoBatch = False
    if X.dim() == 2:
        NoBatch = True
        X = X.unsqueeze(0)  # shape (1, T, F)

    B, F, T = X.shape
    X_aug = X.clone()

    for b in range(B):
        for _ in range(num_masks):
            # Time masking
            t = random.randint(0, tau)
            t0 = random.randint(0, max(1, T - t))
            X_aug[b, :, t0:t0 + t] = 0

            # Frequency masking
            f = random.randint(0, nu)
            f0 = random.randint(0, max(1, F - f))
            X_aug[b, f0:f0 + f, :] = 0

    return X_aug.squeeze(0) if NoBatch else X_aug
