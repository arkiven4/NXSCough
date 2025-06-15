import os
import sys
import glob
import gc
import math
import json
import argparse
import logging
import random
import string
import subprocess
import matplotlib.pyplot as plt

import numpy as np
from scipy.io.wavfile import read
from sklearn.metrics import f1_score, accuracy_score
import librosa

import torch
from torchaudio import transforms as T
from tensorboard.backend.event_processing import event_accumulator

MATPLOTLIB_FLAG = False

logging.basicConfig(stream=sys.stdout, level=logging.ERROR)
logger = logging

def CCC_loss(pred, lab, m_lab=None, v_lab=None, is_numpy=False):
    """
    pred: (N, 3)
    lab: (N, 3)
    """
    if is_numpy:
        pred = torch.Tensor(pred).float().cuda()
        lab = torch.Tensor(lab).float().cuda()
    
    m_pred = torch.mean(pred, 0, keepdim=True)
    m_lab = torch.mean(lab, 0, keepdim=True)

    d_pred = pred - m_pred
    d_lab = lab - m_lab

    v_pred = torch.var(pred, 0, unbiased=False)
    v_lab = torch.var(lab, 0, unbiased=False)

    corr = torch.sum(d_pred * d_lab, 0) / (torch.sqrt(torch.sum(d_pred ** 2, 0)) * torch.sqrt(torch.sum(d_lab ** 2, 0)))

    s_pred = torch.std(pred, 0, unbiased=False)
    s_lab = torch.std(lab, 0, unbiased=False)

    ccc = (2*corr*s_pred*s_lab) / (v_pred + v_lab + (m_pred[0]-m_lab[0])**2)    
    return ccc

def CE_weight_category(pred, lab, weights, test=False):
    if test == False:
      criterion = torch.nn.CrossEntropyLoss(weight=weights)
      loss = criterion(pred, lab)
      
      # Convert logits to predicted class indices
      pred_labels = torch.argmax(pred, dim=1).cpu().numpy()
      lab = lab.cpu().numpy()

      # Compute F1 scores and accuracy
      f1_micro = f1_score(lab, pred_labels, average='micro')
      f1_macro = f1_score(lab, pred_labels, average='macro')
      accuracy = accuracy_score(lab, pred_labels)

      return loss, f1_micro, f1_macro, accuracy
    else:
      # Convert logits to predicted class indices
      pred_labels = pred.cpu().numpy()
      lab = lab.cpu().numpy()

      # Compute F1 scores and accuracy
      f1_micro = f1_score(lab, pred_labels, average='micro')
      f1_macro = f1_score(lab, pred_labels, average='macro')
      accuracy = accuracy_score(lab, pred_labels)
      return f1_micro, f1_macro, accuracy

def warm_start_model(checkpoint_path, model, ignore_layers):
    assert os.path.isfile(checkpoint_path)
    print("Warm starting model from checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model_dict = checkpoint_dict['model']

    random_weight_layer = []
    mismatched_layers = []
    unfound_layers = []
    
    for key, value in model_dict.items(): # model_dict warmstart weight
        if hasattr(model, 'module'): # model is current model
            if key in model.module.state_dict() and value.size() != model.module.state_dict()[key].size():
                try:
                    model_dict[key] = transfer_weight(model_dict[key], model.module.state_dict()[key].size())
                    if model_dict[key].size() != model.module.state_dict()[key].size():
                      mismatched_layers.append(key)
                    else:
                      random_weight_layer.append(key)
                except:
                    mismatched_layers.append(key)
        else:
            if key in model.state_dict() and value.size() != model.state_dict()[key].size():
                try:
                    model_dict[key] = transfer_weight(model_dict[key], model.state_dict()[key].size())
                    if model_dict[key].size() != model.state_dict()[key].size():
                      mismatched_layers.append(key)
                    else:
                      random_weight_layer.append(key)
                except:
                    mismatched_layers.append(key)
        
    print("Mismatched")
    print(mismatched_layers)

    print("random_weight_layer")
    print(random_weight_layer)
    
    ignore_layers = ignore_layers + mismatched_layers
    if len(ignore_layers) > 0:
        model_dict = {k: v for k, v in model_dict.items()
                      if k not in ignore_layers}
        if hasattr(model, 'module'):
          dummy_dict = model.module.state_dict()
          dummy_dict.update(model_dict)
        else:
          dummy_dict = model.state_dict()
          dummy_dict.update(model_dict)
        model_dict = dummy_dict

    if hasattr(model, 'module'):
      model.module.load_state_dict(model_dict, strict=False)
    else:
      model.load_state_dict(model_dict, strict=False)

    return model

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
  assert os.path.isfile(checkpoint_path)
  checkpoint_dict = torch.load(checkpoint_path, weights_only=True, map_location='cpu')
  iteration = 1
  if 'iteration' in checkpoint_dict.keys():
    iteration = checkpoint_dict['iteration']
  if 'learning_rate' in checkpoint_dict.keys():
    learning_rate = checkpoint_dict['learning_rate']
  if optimizer is not None and 'optimizer' in checkpoint_dict.keys():
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
  if scheduler is not None and 'scheduler' in checkpoint_dict.keys():
    scheduler.load_state_dict(checkpoint_dict['scheduler'])
  saved_state_dict = checkpoint_dict['model']
  if hasattr(model, 'module'):
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()
  new_state_dict= {}
  for k, v in state_dict.items():
    try:
      new_state_dict[k] = saved_state_dict[k]
    except:
      print("%s is not in the checkpoint" % k)
      new_state_dict[k] = v
  if hasattr(model, 'module'):
    model.module.load_state_dict(new_state_dict)
  else:
    model.load_state_dict(new_state_dict)
  logger.info("Loaded checkpoint '{}' (iteration {})" .format(
    checkpoint_path, iteration))

  return model, optimizer, scheduler, learning_rate, iteration


def save_checkpoint(model, optimizer, scheduler, learning_rate, iteration, checkpoint_path):
  logger.info("Saving model and optimizer state at iteration {} to {}".format(
    iteration, checkpoint_path))
  if hasattr(model, 'module'):
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()
  torch.save({'model': state_dict,
              'iteration': iteration,
              'optimizer': optimizer.state_dict(),
              'scheduler': scheduler.state_dict(),
              'learning_rate': learning_rate}, checkpoint_path)

def latest_checkpoint_path(dir_path, regex="G_*.pth"):
  f_list = glob.glob(os.path.join(dir_path, regex))
  f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
  x = f_list[-1]
  print(x)
  return x

def get_logger(model_dir, filename="train.log"):
  global logger
  logger = logging.getLogger(os.path.basename(model_dir))
  logger.setLevel(logging.DEBUG)
  
  formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
  if not os.path.exists(model_dir):
    os.makedirs(model_dir)
  h = logging.FileHandler(os.path.join(model_dir, filename))
  h.setLevel(logging.DEBUG)
  h.setFormatter(formatter)
  logger.addHandler(h)
  return logger

def plot_emocoor_to_numpy(pitch, pitch_pred):
  global MATPLOTLIB_FLAG
  if not MATPLOTLIB_FLAG:
    import matplotlib
    matplotlib.use("Agg")
    MATPLOTLIB_FLAG = True
    mpl_logger = logging.getLogger('matplotlib')
    mpl_logger.setLevel(logging.WARNING)

  import matplotlib.pylab as plt
  import numpy as np
  
  fig, ax = plt.subplots()
  ax.scatter(pitch[:, 0], pitch[:, 1], label="Original")
  ax.scatter(pitch_pred[:, 0], pitch_pred[:, 1], label="Prediction")
  plt.tight_layout()
  plt.legend(loc="upper left")

  fig.canvas.draw()
  data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
  data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  plt.close()
  return data

def summarize(writer, global_step, scalars={}, histograms={}, images={}):
  for k, v in scalars.items():
    writer.add_scalar(k, v, global_step)
  for k, v in histograms.items():
    writer.add_histogram(k, v, global_step)
  for k, v in images.items():
    writer.add_image(k, v, global_step, dataformats='HWC')

def load_wav_to_torch(full_path):
  sampling_rate, data = read(full_path)
  return torch.FloatTensor(data.astype(np.float32)), sampling_rate

def get_hparams_from_file(config_path):
  with open(config_path, "r") as f:
    data = f.read()
  config = json.loads(data)

  hparams =HParams(**config)
  return hparams

def get_hparams_from_dir(model_dir):
  config_save_path = os.path.join(model_dir, "config.json")
  with open(config_save_path, "r") as f:
    data = f.read()
  config = json.loads(data)

  hparams =HParams(**config)
  hparams.model_dir = model_dir
  return hparams

class HParams():
  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      if type(v) == dict:
        v = HParams(**v)
      self[k] = v
    
  def keys(self):
    return self.__dict__.keys()

  def items(self):
    return self.__dict__.items()

  def values(self):
    return self.__dict__.values()

  def __len__(self):
    return len(self.__dict__)

  def __getitem__(self, key):
    return getattr(self, key)

  def __setitem__(self, key, value):
    return setattr(self, key, value)

  def __contains__(self, key):
    return key in self.__dict__

  def __repr__(self):
    return self.__dict__.__repr__()

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

def load_audio_sample(file_path, db_sample_rate, wav_stats, desired_length, fade_samples_ratio=6, pad_types='zero'):
    data, sample_rate = librosa.load(file_path, sr=None)
    if sample_rate != db_sample_rate:
        raise ValueError("{} {} SR doesn't match target {} SR".format(sample_rate, db_sample_rate))
    
    #data = data / 32768.0
    max_val = np.max(np.abs(data))
    data = data / max_val if max_val != 0 else data

    data = torch.from_numpy(data).unsqueeze(0)

    fade_samples = int(sample_rate / fade_samples_ratio)
    fade = T.Fade(fade_in_len=fade_samples, fade_out_len=fade_samples, fade_shape='linear')
    data = fade(data)
    data = cut_pad_sample_torchaudio(data, sample_rate, desired_length, pad_types=pad_types)
    return data

def smoothing_tensorboard(values, weight=0.9):
    """EMA smoothing of a curve."""
    smoothed = []
    last = values[0]
    for val in values:
        smoothed_val = last * weight + (1 - weight) * val
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def extract_scalar(log_dir, tag='loss/g/total'):
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    return ea.Scalars(tag)

def plot_loss_from_tensorboard(best_lost, train_log_dir, val_log_dir, tag='loss/g/total', save_path='loss_plot.png'):
    train_events = extract_scalar(train_log_dir, tag)
    val_events = extract_scalar(val_log_dir, tag)

    stop_step = 9999999
    steps_val = []
    values_val = []
    for event in val_events:
        steps_val.append(event.step)
        values_val.append(event.value)
        if event.value == best_lost:
            stop_step = event.step
            break

    steps_train = []
    values_train = []
    for e in train_events:
        if e.step > stop_step:
            break
        steps_train.append(e.step)
        values_train.append(e.value)

    values_train = smoothing_tensorboard(values_train, weight=0.3)
    values_val = smoothing_tensorboard(values_val, weight=0.3)

    plt.figure(figsize=(10, 3))
    plt.plot(steps_train, values_train, ':k', label='Train')
    plt.plot(steps_val, values_val, '-k', label='Validation')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title(f'Loss Training')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()