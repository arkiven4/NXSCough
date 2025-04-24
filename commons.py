
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

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