import argparse, gc, glob, json, logging, math, os, random, socket, string, subprocess, sys, hashlib, pickle
import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
from scipy.io.wavfile import read
from sklearn.metrics import accuracy_score, f1_score
from tensorboard.backend.event_processing import event_accumulator
from torchaudio import transforms as T
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from scipy.signal import resample
from tqdm import tqdm

import losses
MATPLOTLIB_FLAG = False

logging.basicConfig(stream=sys.stdout, level=logging.ERROR)
logger = logging

DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3

def piecewise_rational_quadratic_transform(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    tails=None,
    tail_bound=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
):
    if tails is None:
        spline_fn = rational_quadratic_spline
        spline_kwargs = {}
    else:
        spline_fn = unconstrained_rational_quadratic_spline
        spline_kwargs = {"tails": tails, "tail_bound": tail_bound}

    outputs, logabsdet = spline_fn(
        inputs=inputs,
        unnormalized_widths=unnormalized_widths,
        unnormalized_heights=unnormalized_heights,
        unnormalized_derivatives=unnormalized_derivatives,
        inverse=inverse,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative,
        **spline_kwargs,
    )
    return outputs, logabsdet


def searchsorted(bin_locations, inputs, eps=1e-6):
    bin_locations[..., -1] += eps
    return torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1


def unconstrained_rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    tails="linear",
    tail_bound=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
):
    inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside_interval_mask = ~inside_interval_mask

    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)

    if tails == "linear":
        unnormalized_derivatives = F.pad(unnormalized_derivatives, pad=(1, 1))
        constant = np.log(np.exp(1 - min_derivative) - 1)
        unnormalized_derivatives[..., 0] = constant
        unnormalized_derivatives[..., -1] = constant

        outputs[outside_interval_mask] = inputs[outside_interval_mask]
        logabsdet[outside_interval_mask] = 0
    else:
        raise RuntimeError("{} tails are not implemented.".format(tails))

    outputs[inside_interval_mask], logabsdet[inside_interval_mask] = rational_quadratic_spline(
        inputs=inputs[inside_interval_mask],
        unnormalized_widths=unnormalized_widths[inside_interval_mask, :],
        unnormalized_heights=unnormalized_heights[inside_interval_mask, :],
        unnormalized_derivatives=unnormalized_derivatives[inside_interval_mask, :],
        inverse=inverse,
        left=-tail_bound,
        right=tail_bound,
        bottom=-tail_bound,
        top=tail_bound,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative,
    )

    return outputs, logabsdet


def rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    left=0.0,
    right=1.0,
    bottom=0.0,
    top=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
):
    if torch.min(inputs) < left or torch.max(inputs) > right:
        raise ValueError("Input to a transform is not within its domain")

    num_bins = unnormalized_widths.shape[-1]

    if min_bin_width * num_bins > 1.0:
        raise ValueError("Minimal bin width too large for the number of bins")
    if min_bin_height * num_bins > 1.0:
        raise ValueError("Minimal bin height too large for the number of bins")

    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode="constant", value=0.0)
    cumwidths = (right - left) * cumwidths + left
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = min_derivative + F.softplus(unnormalized_derivatives)

    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), mode="constant", value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    if inverse:
        bin_idx = searchsorted(cumheights, inputs)[..., None]
    else:
        bin_idx = searchsorted(cumwidths, inputs)[..., None]

    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]

    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    delta = heights / widths
    input_delta = delta.gather(-1, bin_idx)[..., 0]

    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]

    input_heights = heights.gather(-1, bin_idx)[..., 0]

    if inverse:
        a = (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        ) + input_heights * (input_delta - input_derivatives)
        b = input_heights * input_derivatives - (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        )
        c = -input_delta * (inputs - input_cumheights)

        discriminant = b.pow(2) - 4 * a * c
        assert (discriminant >= 0).all()

        root = (2 * c) / (-b - torch.sqrt(discriminant))
        outputs = root * input_bin_widths + input_cumwidths

        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta) * theta_one_minus_theta
        )
        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * root.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - root).pow(2)
        )
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

        return outputs, -logabsdet
    else:
        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)

        numerator = input_heights * (input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta)
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta) * theta_one_minus_theta
        )
        outputs = input_cumheights + numerator / denominator

        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * theta.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - theta).pow(2)
        )
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

        return outputs, logabsdet

def get_free_port():
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port

def stratified_group_split(df, label_col='tb_status', group_col='participant', test_size=0.2, random_state=42):
    sgkf = StratifiedGroupKFold(n_splits=int(1/test_size), shuffle=True, random_state=random_state)
    for train_idx, val_idx in sgkf.split(df, df[label_col], df[group_col]):
        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_val = df.iloc[val_idx].reset_index(drop=True)
        return df_train, df_val

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

def seed_from_path(path):
    h = hashlib.md5(path.encode()).hexdigest()
    seed = int(h[:8], 16) 
    random.seed(seed)
    np.random.seed(seed)

def cut_pad_sample_torchaudio(data, sample_rate, desired_length, pad_types='zero', right_pad_shift_sec=0.04):
    fade_samples_ratio = 6
    fade_samples = int(sample_rate / fade_samples_ratio)
    fade_out = T.Fade(fade_in_len=0, fade_out_len=fade_samples, fade_shape='linear')
    target_duration = int(desired_length * sample_rate)
    current_length = data.shape[-1]

    if data.shape[-1] > target_duration:
        max_start = data.shape[-1] - target_duration
        start = np.random.randint(0, max_start + 1)  # random start index
        data = data[..., start:start + target_duration]
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


def apply_fade_in(audio, sr, sec=0.08):
    fade_len = int(sr * sec)
    fade_len = min(fade_len, len(audio))
    fade = np.linspace(0.0, 1.0, fade_len)
    audio[:fade_len] *= fade
    return audio

def apply_fade_out(audio, sr, sec=0.08):
    fade_len = int(sr * sec)
    fade_len = min(fade_len, len(audio))
    fade = np.linspace(1.0, 0.0, fade_len)
    audio[-fade_len:] *= fade
    return audio

def speed_perturb(audio, factor_min=0.9, factor_max=1.1):
    factor = random.uniform(factor_min, factor_max)
    new_len = int(len(audio) / factor)
    out = resample(audio, new_len)
    return np.clip(out, -1.0, 1.0)

def gain_perturb(audio, db_range):
    low, high = db_range
    gain_db = random.uniform(low, high)
    gain = 10 ** (gain_db / 20)
    out = audio * gain
    return np.clip(out, -1.0, 1.0)

def generate_gap(sr, low=0.08, high=0.12, noise_gain=0.0001):
    gap_sec = random.uniform(low, high)
    gap_len = int(sr * gap_sec)
    noise = np.random.randn(gap_len) * noise_gain
    return np.clip(noise, -1.0, 1.0)

def build_tail_segment(audio, sr):
    tail_gap = generate_gap(sr, low=0.15, high=0.4)
    tail = audio.copy()

    tail = speed_perturb(tail)
    tail = gain_perturb(tail, (-5.0, -1.0))
    return np.concatenate([tail_gap, tail], axis=0)

def augment_and_merge(audio_original, path, sr, gain_db_set=[(-5.0, -1.0)]):
    seed_from_path(path)

    # fade-out original
    audio_faded = audio_original.copy()
    audio_faded = apply_fade_out(audio_faded, sr)

    # extract segment aligned to energy peak
    sec_i_start = max((audio_original ** 2).argmax() - 2400, 0)
    sec_segment = audio_original.copy()[sec_i_start:]

    # fade-in
    sec_segment = apply_fade_in(sec_segment, sr)

    # speed perturbation
    sec_segment = speed_perturb(sec_segment)

    # gain perturbation
    gain_db = random.choice(gain_db_set)
    sec_segment = gain_perturb(sec_segment, gain_db)

    # noise gap prepend
    gap_seg = generate_gap(sr)
    sec_segment = np.concatenate([gap_seg, sec_segment], axis=0)

    if random.random() < 0.50:
        tail_segment = build_tail_segment(sec_segment, sr)
        sec_segment = np.concatenate([sec_segment, tail_segment], axis=0)

    # merge
    merged = np.concatenate([audio_faded, sec_segment], axis=0)
    return merged


def load_audio_sample(file_path, db_sample_rate, is_saming_length, desired_length, fade_samples_ratio=6, pad_types='zero'):
    data, sample_rate = librosa.load(file_path, sr=db_sample_rate)
    if sample_rate != db_sample_rate:
        raise ValueError("{} SR doesn't match target {} SR".format(sample_rate, db_sample_rate))
    
    if is_saming_length:
      data = torch.from_numpy(data).unsqueeze(0)
      fade_samples = int(sample_rate / fade_samples_ratio)
      fade = T.Fade(fade_in_len=fade_samples, fade_out_len=fade_samples, fade_shape='linear')
      data = fade(data)
      data = cut_pad_sample_torchaudio(data, sample_rate, desired_length, pad_types=pad_types)

    if pad_types == "synthesis":
      data = augment_and_merge(data, path=file_path, sr=sample_rate)
      
    return data if torch.is_tensor(data) else torch.from_numpy(data).unsqueeze(0)

def plot_spectrogram_to_numpy(spectrogram):
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
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                    interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)

    w, h = fig.canvas.get_width_height()
    buf = buf.reshape((h, w, 4))  # ARGB

    # Convert ARGB → RGB
    buf = buf[:, :, 1:]  # drop alpha, keep R G B

    plt.close()
    return torch.from_numpy(buf.copy()).permute(2, 0, 1).float() / 255.0

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

def orthogonality_loss(d_emb, s_emb):
    # penalize correlation between disease and speaker embedding per-sample
    # normalize to remove scale effect
    d = F.normalize(d_emb, p=2, dim=1)
    s = F.normalize(s_emb, p=2, dim=1)
    # dot product per sample then squared mean
    dots = torch.sum(d * s, dim=1)  # (B,)
    return torch.mean(dots * dots)

def compute_class_weights(df, target_col, device="cuda"):
    disease_codes = df[target_col].unique().tolist()
    class_frequencies = df[target_col].value_counts().to_dict()
    total_samples = len(df)

    class_weights = {
        cls: total_samples / (len(disease_codes) * freq) if freq != 0 else 0
        for cls, freq in class_frequencies.items()
    }

    weights_list = [class_weights[cls] for cls in disease_codes]
    return torch.tensor(weights_list, device=device, dtype=torch.float)


def compute_wav_stats(df, path_col, pickle_path="wav_stats.pickle"):
    if os.path.exists(pickle_path):
        with open(pickle_path, "rb") as f:
            return pickle.load(f)

    means, stds = [], []
    paths = df[path_col].dropna().tolist()

    for path in tqdm(paths, desc="Processing WAV files", unit="file"):
        if not os.path.isfile(path):
            continue
        try:
            audio, _ = librosa.load(path, sr=None, mono=True)
            means.append(np.mean(audio))
            stds.append(np.std(audio))
        except Exception:
            continue

    stats = {
        "mean_db": float(np.mean(means)),
        "std_db": float(np.mean(stds))
    }

    with open(pickle_path, "wb") as f:
        pickle.dump(stats, f)

    return stats

def many_loss_category(pred, lab, loss_type="CE", test=False, weights=None):
    if test == True:
        pred_labels = torch.max(pred, 1).cpu().numpy()
        lab = lab.cpu().numpy()

        f1_micro = f1_score(lab, pred_labels, average='micro')
        f1_macro = f1_score(lab, pred_labels, average='macro')
        accuracy = accuracy_score(lab, pred_labels)
        return f1_micro, f1_macro, accuracy

    if loss_type == "CE":
        criterion = torch.nn.CrossEntropyLoss(weight=weights)
        loss = criterion(pred, lab)
        return [loss]
    elif loss_type == "BCE":
        criterion = torch.nn.BCEWithLogitsLoss(weight=weights)
        loss = criterion(pred, lab)
        return [loss]
    elif loss_type == "KLDivLoss":
        criterion = losses.KLDivLoss()
        loss = criterion(pred, lab)
        return [loss]