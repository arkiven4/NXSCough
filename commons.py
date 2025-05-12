
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from librosa.filters import mel as librosa_mel_fn
from audio_processing import dynamic_range_compression, dynamic_range_decompression
from stft import STFT

from ifas import extract_IFAS

def compute_length_from_mask(mask, frame_shift=0.02):
    """
    mask: (batch_size, T)
    Assuming that the sampling rate is 16kHz, the frame shift is 20ms
    """
    wav_lens = torch.sum(mask, dim=1) # (batch_size, )
    feat_lens = torch.div(wav_lens-1, 16000*frame_shift, rounding_mode="floor") + 1
    feat_lens = feat_lens.int().tolist()
    return feat_lens

def sequence_mask(length, max_length=None):
  if max_length is None:
    max_length = length.max()
  x = torch.arange(max_length, dtype=length.dtype, device=length.device)
  return x.unsqueeze(0) < length.unsqueeze(1)

def convert_pad_shape(pad_shape):
  l = pad_shape[::-1]
  pad_shape = [item for sublist in l for item in sublist]
  return pad_shape

def shift_1d(x):
  x = F.pad(x, convert_pad_shape([[0, 0], [0, 0], [1, 0]]))[:, :, :-1]
  return x

def clip_grad_value_(parameters, clip_value, norm_type=2):
  if isinstance(parameters, torch.Tensor):
    parameters = [parameters]
  parameters = list(filter(lambda p: p.grad is not None, parameters))
  norm_type = float(norm_type)
  if clip_value is not None:
    clip_value = float(clip_value)

  total_norm = 0
  for p in parameters:
    param_norm = p.grad.data.norm(norm_type)
    total_norm += param_norm.item() ** norm_type

    if clip_value is not None:
      p.grad.data.clamp_(min=-clip_value, max=clip_value)
  total_norm = total_norm ** (1. / norm_type)
  return total_norm

class TacotronSTFT(nn.Module):
  def __init__(self, filter_length=1024, hop_length=256, win_length=1024,
                 n_mel_channels=80, sampling_rate=22050, mel_fmin=0.0,
                 mel_fmax=8000.0):
    super(TacotronSTFT, self).__init__()
    self.n_mel_channels = n_mel_channels
    self.sampling_rate = sampling_rate
    self.stft_fn = STFT(filter_length, hop_length, win_length)
    mel_basis = librosa_mel_fn(
        sr=sampling_rate, n_fft=filter_length, n_mels=n_mel_channels, fmin=mel_fmin, fmax=mel_fmax)
    mel_basis = torch.from_numpy(mel_basis).float()
    self.register_buffer('mel_basis', mel_basis)

  def spectral_normalize(self, magnitudes):
    output = dynamic_range_compression(magnitudes)
    return output

  def spectral_de_normalize(self, magnitudes):
    output = dynamic_range_decompression(magnitudes)
    return output

  def forward(self, y):
    """Computes mel-spectrograms from a batch of waves
    PARAMS
    ------
    y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]

    RETURNS
    -------
    mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
    """
    y = y.unsqueeze(0)
    assert(torch.min(y.data) >= -1)
    assert(torch.max(y.data) <= 1)

    magnitudes, phases = self.stft_fn.transform(y)
    magnitudes = magnitudes.data
    mel_output = torch.matmul(self.mel_basis, magnitudes)
    mel_output = self.spectral_normalize(mel_output).squeeze(0)
    return mel_output
  
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class IFASv2(nn.Module):
  def __init__(self, fs_sca, K_sca=2048, K_max=160, B_sca=512):
    super(IFASv2, self).__init__()
    self.fs_sca = fs_sca
    self.K_sca = K_sca
    self.K_max = K_max
    self.B_sca = B_sca

  def spectral_normalize(self, magnitudes):
    output = dynamic_range_compression(magnitudes)
    return output

  def forward(self, x_vec):
    """Computes mel-spectrograms from a batch of waves
    PARAMS
    ------
    y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]

    RETURNS
    -------
    mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
    """
    path_metadata = x_vec.split("/")
    file_metadata = path_metadata[-1].split(".wav")[0]
    input_path = "/".join(path_metadata[:-1]) + "/"
    input_path = input_path.replace("CombineData/", "Extracted_Feature/IFAS/")

    if_mat = np.load(f'{input_path}/{file_metadata}.npy')
    if_mat[np.isnan(if_mat)] = 1e-8
    if_mat = np.log(if_mat)
    if_mat = torch.from_numpy(if_mat)

    # x_vec = x_vec.numpy()
    # if_mat = extract_IFAS(x_vec, self.fs_sca, K_sca=self.K_sca, B_sca=self.B_sca) 
    # if_mat = if_mat[:self.K_max, :]
    # if_mat[np.isnan(if_mat)] = 1e-8
    # if_mat = torch.from_numpy(if_mat).unsqueeze(0)
    # if_mat = self.spectral_normalize(if_mat).squeeze(0)
    return if_mat

# def load_IFAS(fullpath):
#   path_metadata = fullpath.split("/")
#   file_metadata = path_metadata[-1].split(".wav")[0]
#   input_path = "/".join(path_metadata[:-1]) + "/"
#   input_path = input_path.replace("CombineData/", "Extracted_Feature/IFAS/")

#   if_mat = np.load(f'{input_path}/{file_metadata}.npy')
#   if_mat[np.isnan(if_mat)] = 0
#   epsilon = 1e-8
#   if_mat = np.log(if_mat + epsilon)
#   if_mat = torch.from_numpy(if_mat)

#   return if_mat