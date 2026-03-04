from collections import defaultdict
from torch.utils.data import Sampler
from matplotlib import cm
import os
import pickle
import random, math

import numpy as np
import pandas as pd
from sklearn.utils import shuffle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchaudio import transforms as T
import torchvision.transforms as transforms

import joblib
import librosa
import opensmile
from gammatone.gtgram import gtgram

import utils
import commons
import audio_processing
from augmentation import (
    DataAugmentator,
    ISD_additive_noise,
    LnL_convolutive_noise,
)

FEATURE_SETS = [
    opensmile.FeatureSet.ComParE_2016,
    opensmile.FeatureSet.GeMAPSv01b,
    opensmile.FeatureSet.eGeMAPSv02,
    opensmile.FeatureSet.emobase,
    opensmile.FeatureSet.IS09,
    opensmile.FeatureSet.IS10,
    opensmile.FeatureSet.IS11,
    opensmile.FeatureSet.IS12,
    opensmile.FeatureSet.IS13,
]

SMILE_CLIENTS = {
    str(fs): opensmile.Smile(
        feature_set=fs,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors
    )
    for fs in FEATURE_SETS
}

viridis_lut = torch.tensor(cm.get_cmap("viridis").colors, dtype=torch.float32)  # (256,4)
viridis_lut = viridis_lut[:, :3]


def build_wav_transform(hparams, sampling_rate):
    """Return the appropriate waveform-to-feature transform callable for the given hparams."""
    feature_type = hparams.feature_type

    if feature_type == "mfcc":
        return lambda wav: torch.tensor(
            librosa.feature.mfcc(
                y=wav.numpy() if isinstance(wav, torch.Tensor) else wav,
                sr=sampling_rate,
                n_fft=hparams.filter_length,
                win_length=hparams.win_length,
                hop_length=hparams.hop_length,
                n_mfcc=13,
            ),
            dtype=torch.float32,
        )

    elif feature_type == "chroma":
        return lambda wav: torch.tensor(
            librosa.feature.chroma_stft(
                y=wav.numpy() if isinstance(wav, torch.Tensor) else wav,
                sr=sampling_rate,
                n_fft=hparams.filter_length,
                win_length=hparams.win_length,
                hop_length=hparams.hop_length,
                n_chroma=12,
                tuning=0.0,
            ),
            dtype=torch.float32,
        )

    elif feature_type == "melspectogram":
        return lambda wav: torch.tensor(
            librosa.power_to_db(
                librosa.feature.melspectrogram(
                    y=wav.numpy() if isinstance(wav, torch.Tensor) else wav,
                    sr=hparams.sampling_rate,
                    n_mels=hparams.n_mel_channels,
                    fmin=hparams.mel_fmin,
                    fmax=hparams.mel_fmax,
                    n_fft=hparams.filter_length,
                    hop_length=hparams.hop_length,
                    power=2.0,
                ),
                ref=np.max,
            ),
            dtype=torch.float32,
        )

    elif feature_type == "logmel":
        return lambda wav: torch.tensor(
            np.log(
                librosa.feature.melspectrogram(
                    y=wav.numpy() if isinstance(wav, torch.Tensor) else wav,
                    sr=hparams.sampling_rate,
                    n_mels=hparams.n_mel_channels,
                    n_fft=hparams.filter_length,
                    hop_length=hparams.hop_length,
                    win_length=hparams.win_length,
                ) + 1e-6
            ),
            dtype=torch.float32,
        )

    elif feature_type == "gammmaspectogram":
        return lambda wav: torch.tensor(
            np.log(
                gtgram(
                    wave=wav.numpy() if isinstance(wav, torch.Tensor) else wav,
                    fs=hparams.sampling_rate,
                    channels=hparams.n_mel_channels,
                    f_min=hparams.mel_fmin,
                    f_max=hparams.mel_fmax,
                    window_time=hparams.filter_length / hparams.sampling_rate,
                    hop_time=hparams.hop_length / hparams.sampling_rate,
                ) + 1e-8
            ),
            dtype=torch.float32,
        )

    elif feature_type == "spectogram":
        return lambda wav: torch.tensor(
            librosa.power_to_db(
                np.abs(librosa.stft(
                    y=wav.numpy() if isinstance(wav, torch.Tensor) else wav,
                    n_fft=hparams.filter_length,
                    hop_length=hparams.hop_length,
                )) ** 2,
                ref=np.max,
            ),
            dtype=torch.float32,
        )

    elif feature_type == "fbank_ast":
        from transformers import AutoFeatureExtractor
        feature_extractor_ast = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        return lambda wav: (
            feature_extractor_ast(
                wav.squeeze(0), sampling_rate=sampling_rate, return_tensors="pt"
            )["input_values"][0].T
        )

    elif feature_type == "opensmile":
        scaler = joblib.load("precomputed_stats/opensmile_global_scaler.pkl")
        return lambda wav: torch.tensor(
            scaler.transform(
                pd.concat(
                    [
                        client.process_signal(wav, sampling_rate)
                        .reset_index(drop=True)
                        .add_prefix(f"{fs_name}__")
                        for fs_name, client in SMILE_CLIENTS.items()
                    ],
                    axis=1,
                ).fillna(0.0).values
            ).T,
            dtype=torch.float32,
        )
    return None


class CoughDatasets(torch.utils.data.Dataset):
    def __init__(self, data_numpy, hparams, train=True, wav_stats_path=None, use_precomputed=False):
        self.audiopaths_and_text = data_numpy
        self.train = getattr(hparams, "train", train)
        self.use_precomputed = use_precomputed
        self._init_hparams(hparams)

        if self.augment_data:
            # ["apply_speed_perturbation", "apply_pitch_shift", "apply_reverb", "add_background_noise", "apply_random_gain"]
            self.data_augmentator = DataAugmentator(
                None,
                augmentation_noises_labels_path="/run/media/fourier/Data1/Pras/Interspeech2025/RIRS_NOISES/data_augmentation_noises_speechs_labels.tsv",
                augmentation_rirs_directory=None,
                augmentation_rirs_labels_path="/run/media/fourier/Data1/Pras/Interspeech2025/RIRS_NOISES/data_augmentation_rirs_labels.tsv",
                augmentation_window_size_secs=5.5, augmentation_probability=[0.3, 0.25, 0.2, 0.25, 0.0]) # 0.3, 0.15, 0.2, 0.4, 0.0

        if self.mean_std_norm and not self.use_precomputed:
            with open(wav_stats_path, 'rb') as f:
                stats = pickle.load(f)
                self.wav_stats = stats

        self.wav_transform = None
        if hparams.acoustic_feature and not self.use_precomputed:
            self.wav_transform = build_wav_transform(hparams, self.sampling_rate)

    def _init_hparams(self, hparams):
        """Unpack all hparams fields into instance attributes."""
        self.hop_length = getattr(hparams, "hop_length", None)
        self.max_wav_value = getattr(hparams, "max_wav_value", None)
        self.mean_std_norm = getattr(hparams, "mean_std_norm", False)
        self.cmvn_norm = getattr(hparams, "cmvn_norm", False)
        self.per_band_norm = getattr(hparams, "per_band_norm", False)
        self.sampling_rate = getattr(hparams, "sampling_rate", None)
        self.saming_length = getattr(hparams, "saming_length", None)
        self.desired_length = getattr(hparams, "desired_length", None)
        self.pad_types = getattr(hparams, "pad_types", None)
        self.feature_type = getattr(hparams, "feature_type", None)
        self.delta_feature = getattr(hparams, "delta_feature", None)
        self.deltadelta_feature = getattr(hparams, "deltadelta_feature", None)
        self.db_path = getattr(hparams, "db_path", None)

        self.tabular_feature = getattr(hparams, "tabular_feature", False)
        self.augment_data = getattr(hparams, "augment_data", False)
        self.augment_rawboost = getattr(hparams, "augment_rawboost", False)
        self.augment_prob = getattr(hparams, "augment_prob", 0.6)
        self.multimask_augment = getattr(hparams, "multimask_augment", False)
        self.multimask_prob = getattr(hparams, "multimask_prob", False)
        self.fade_samples_ratio = getattr(hparams, "fade_samples_ratio", 0.0)
        self.add_noise = getattr(hparams, "add_noise", False)
        self.mix_audio = getattr(hparams, "mix_audio", False)

        self.tau = getattr(hparams, "tau", 0.0)
        self.nu = getattr(hparams, "nu", 0.0)
        self.num_masks = getattr(hparams, "num_masks", 0)
        self.mae_training = getattr(hparams, "mae_training", False)
        self.ssccl_training = getattr(hparams, "ssccl_training", False)
        self.rezize_size = tuple(getattr(hparams, "rezize_size", [224, 224]))

        self.nClasses = getattr(hparams, "many_class", None)
        self.cough_detection = getattr(hparams, "cough_detection", False)
        self.processor = None

    # TODO: Move to INIT
    def set_feature_path_column(self, col_index):
        """Set the column index where precomputed feature paths are stored."""
        self.feature_path_col = col_index

    def get_mel_text_pair(self, audiopath_and_text):
        wavname, dse_id, gndr_id, spk_id = audiopath_and_text[0], audiopath_and_text[1], audiopath_and_text[2], audiopath_and_text[3]
        if self.use_precomputed and hasattr(self, 'feature_path_col'):
            wavname = audiopath_and_text[self.feature_path_col] 

        wav1, dse_id = self.get_audio(wavname, dse_id)

        tabular = np.zeros((1, 4))
        if self.tabular_feature:
            tabular = torch.from_numpy(audiopath_and_text[4:8].astype("float32")).reshape(1, -1)

        return (wavname, wav1, dse_id, int(spk_id), int(gndr_id), tabular)

    def get_audio(self, filename, dse_id=None, always_augment=False, start_index=0, end_index=-1):
        dse_id = int(dse_id)
        eye = torch.eye(self.nClasses, dtype=torch.float32)
        dse_id = eye[dse_id]
        dse_id = dse_id.unsqueeze(0)
    
        if self.use_precomputed and hasattr(self, 'feature_path_col'):
            audio = torch.load(filename)
        else:
            audio = utils.load_audio_sample(filename, self.sampling_rate, self.saming_length, self.desired_length, 
                                            fade_samples_ratio=self.fade_samples_ratio, pad_types=self.pad_types, train=self.train)  # repeat zero
            audio = audio.squeeze(0)
            if self.train:
                if self.augment_data:
                    if random.random() < self.augment_prob or always_augment:
                        audio = self.data_augmentator(audio.unsqueeze(0), self.sampling_rate).squeeze(0)

                if self.augment_rawboost:
                    x = audio.numpy() if isinstance(audio, torch.Tensor) else audio
                    x = LnL_convolutive_noise( x, 5, 5, 20, 8000, 100, 1000, 10, 100, 0, 0, 5, 20, self.sampling_rate)
                    x = ISD_additive_noise(x, 10, 2)
                    audio = torch.as_tensor(x, dtype=audio.dtype)

                if self.add_noise:
                    audio = audio + torch.rand_like(audio)

            if self.wav_transform is not None:
                audio = self.wav_transform(audio)  # [80, 224]
                audio = [audio.detach().cpu().numpy()]
                if self.delta_feature:
                    audio.append(librosa.feature.delta(audio[0]))
                if self.deltadelta_feature:
                    audio.append(librosa.feature.delta(audio[0], order=2))
                audio = np.concatenate(audio, axis=0)
                audio = torch.from_numpy(audio).float()

            if self.mean_std_norm and hasattr(self, 'wav_stats'):
                audio = (audio - self.wav_stats['mean_db']) / (self.wav_stats['std_db'] + 1e-6)

            if self.cmvn_norm:
                if audio.ndim >= 2:
                    audio = (audio - audio.mean(dim=1, keepdim=True)) / (audio.std(dim=1, keepdim=True, unbiased=False) + 1e-16)
                else:
                    audio = (audio - audio.mean()) / (audio.std(unbiased=False) + 1e-16)

            if self.max_wav_value:
                max_val = torch.max(torch.abs(audio))
                audio = audio / max_val if max_val != 0 else audio

            # audio torch.Size([80, 32])

        if self.multimask_augment and random.random() < self.multimask_prob and self.train:
            audio = audio_processing.multi_mask_spectrogram(
                audio, tau=int(audio.shape[1] * self.tau),
                nu=int(audio.shape[0] * self.nu),
                num_masks=self.num_masks)  # T, F

        audio = audio.unsqueeze(0)
        return audio, dse_id

    def __getitem__(self, index):
        data = self.get_mel_text_pair(self.audiopaths_and_text[index])
        return (index,) + data

    def __len__(self):
        return len(self.audiopaths_and_text)


class CoughDatasetsCollate:
    def __init__(self, many_data=2, many_tabular=4, processor=None, sampling_rate=16000):
        self.processor = processor
        self.sampling_rate = sampling_rate
        self.many_data = many_data
        self.many_tabular = many_tabular

    def __call__(self, batch):
        lengths = torch.tensor([x[2].shape[-1] for x in batch])
        lengths_sorted, idx = torch.sort(lengths, descending=True)

        max_len = lengths_sorted[0]
        bsz = len(batch)

        first_wav = batch[0][2]
        feature_dim = first_wav.shape[1] if first_wav.ndim == 3 else 1

        sorted_indices = torch.zeros(bsz, dtype=torch.long)
        wav_names = []
        wav_padded = torch.zeros(bsz, feature_dim, lengths_sorted[0], dtype=first_wav.dtype)
        attention_masks = torch.zeros(bsz, max_len)

        dse_ids = torch.zeros(bsz, self.many_data, dtype=torch.float32)
        spk_ids = torch.zeros(bsz, dtype=torch.long)
        gndr_ids = torch.zeros(bsz, dtype=torch.long)
        tabular_ids = torch.zeros(bsz, self.many_tabular, dtype=torch.float32)
        for i, j in enumerate(idx):
            sample_idx, name, wav, dse, spk, gndr, tblr = batch[j]
            wav_names.append(name)
            sorted_indices[i] = sample_idx

            wav_padded[i, :, :wav.shape[-1]] = wav
            attention_masks[i, :wav.shape[-1]] = 1

            dse_ids[i] = dse
            spk_ids[i] = spk
            gndr_ids[i] = gndr
            tabular_ids[i] = tblr.squeeze(0)

        return wav_names, wav_padded, None, attention_masks, dse_ids, [spk_ids, gndr_ids, tabular_ids, sorted_indices]


class CoughDatasetsProcessorCollate():
    """ Zero-pads model inputs and targets based on number of frames per step
    """

    def __init__(self, many_data=2, processor=None, sampling_rate=16000):
        self.processor = processor
        self.sampling_rate = sampling_rate
        self.many_data = many_data

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        (Index, wavname, wav1, wav2, dse_id, int(spk_id), int(gndr_id), tabular)
        """
        wav_name = [x[1] for x in batch]
        wavs = [x[2].squeeze().numpy() for x in batch]  # list[np.ndarray]
        dse_ids = torch.stack([x[4].squeeze(0) for x in batch])
        spk_ids = torch.stack([torch.tensor(x[5]) for x in batch])
        gndr_ids = torch.stack([torch.tensor(x[6]) for x in batch])

        audio_inputs = self.processor.feature_extractor(
            wavs,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
            # padding="max_length",
            truncation=False,  # ?
            return_attention_mask=True,
            padding=True
        )
        wav_padded = audio_inputs["input_features"]           # [B, n_mels, T]
        attention_masks = audio_inputs["attention_mask"]      # [B, T]

        B, n_mels, T = wav_padded.shape
        dummy_wav = torch.empty(
            (B, n_mels, T),
            dtype=wav_padded.dtype
        )

        return wav_name, wav_padded, dummy_wav, attention_masks, dse_ids, [spk_ids, None, gndr_ids, None]


class CoughDetectionRatioBatchSampler(Sampler):
    def __init__(
        self,
        cough_idx,
        speech_idx,
        noise_idx,
        batch_size,
        ratios=(0.5, 0.3, 0.2),
        drop_last=True,
    ):
        self.cough_idx = cough_idx
        self.speech_idx = speech_idx
        self.noise_idx = noise_idx
        self.batch_size = batch_size
        self.ratios = ratios
        self.drop_last = drop_last

        self.n_cough = int(batch_size * ratios[0])
        self.n_speech = int(batch_size * ratios[1])
        self.n_noise = batch_size - self.n_cough - self.n_speech

    def __iter__(self):
        random.shuffle(self.cough_idx)
        random.shuffle(self.speech_idx)
        random.shuffle(self.noise_idx)

        ptr_c, ptr_s, ptr_n = 0, 0, 0

        while True:
            if (
                ptr_c + self.n_cough > len(self.cough_idx)
                or ptr_s + self.n_speech > len(self.speech_idx)
                or ptr_n + self.n_noise > len(self.noise_idx)
            ):
                if self.drop_last:
                    break
                else:
                    ptr_c = ptr_s = ptr_n = 0

            batch = (
                self.cough_idx[ptr_c:ptr_c + self.n_cough]
                + self.speech_idx[ptr_s:ptr_s + self.n_speech]
                + self.noise_idx[ptr_n:ptr_n + self.n_noise]
            )

            random.shuffle(batch)

            ptr_c += self.n_cough
            ptr_s += self.n_speech
            ptr_n += self.n_noise

            yield batch

    def __len__(self):
        return min(
            len(self.cough_idx) // self.n_cough,
            len(self.speech_idx) // self.n_speech,
            len(self.noise_idx) // self.n_noise,
        )


class CoughDiseaseBinaryBatchSampler(Sampler):
    def __init__(
        self,
        positive_idx,
        negative_idx,
        batch_size,
        ratio=0.5,
        drop_last=True,
    ):
        """
        Binary batch sampler for balanced 50/50 sampling.

        Args:
            positive_idx: List of indices for positive class (e.g., cough)
            negative_idx: List of indices for negative class (e.g., non-cough)
            batch_size: Total batch size
            ratio: Ratio of positive samples in batch (default: 0.5 for 50/50)
            drop_last: Whether to drop the last incomplete batch
        """
        self.positive_idx = positive_idx
        self.negative_idx = negative_idx
        self.batch_size = batch_size
        self.ratio = ratio
        self.drop_last = drop_last

        self.n_positive = int(batch_size * ratio)
        self.n_negative = batch_size - self.n_positive

    def __iter__(self):
        random.shuffle(self.positive_idx)
        random.shuffle(self.negative_idx)

        ptr_pos, ptr_neg = 0, 0

        while True:
            if (
                ptr_pos + self.n_positive > len(self.positive_idx)
                or ptr_neg + self.n_negative > len(self.negative_idx)
            ):
                if self.drop_last:
                    break
                else:
                    # Reset pointers and reshuffle
                    random.shuffle(self.positive_idx)
                    random.shuffle(self.negative_idx)
                    ptr_pos = ptr_neg = 0

            batch = (
                self.positive_idx[ptr_pos:ptr_pos + self.n_positive]
                + self.negative_idx[ptr_neg:ptr_neg + self.n_negative]
            )

            random.shuffle(batch)

            ptr_pos += self.n_positive
            ptr_neg += self.n_negative

            yield batch

    def __len__(self):
        return min(
            len(self.positive_idx) // self.n_positive,
            len(self.negative_idx) // self.n_negative,
        )


class PatientBatchSampler(Sampler):
    def __init__(
        self,
        patient_ids,
        patients_per_batch,
        coughs_per_patient,
        shuffle=True,
        drop_last=True,
    ):
        self.patient_ids = patient_ids
        self.P = patients_per_batch
        self.K = coughs_per_patient
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.pid2idx = defaultdict(list)
        for idx, pid in enumerate(patient_ids):
            self.pid2idx[pid].append(idx)

        self.patients = list(self.pid2idx.keys())

    def __iter__(self):
        patients = self.patients.copy()
        if self.shuffle:
            random.shuffle(patients)

        batch = []
        selected_patients = []

        for pid in patients:
            idxs = self.pid2idx[pid]
            if len(idxs) < self.K:
                continue

            batch.extend(random.sample(idxs, self.K))
            selected_patients.append(pid)

            if len(selected_patients) == self.P:
                yield batch
                batch = []
                selected_patients = []

        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        return len(self.patients) // self.P


class AutoPatientBatchSampler(Sampler):
    def __init__(
        self,
        patient_ids,
        labels,
        target_batch_wavs=128,
        pos_fraction=0.5,
        shuffle=True,
    ):
        self.patient_ids = patient_ids
        self.labels = labels
        self.target_batch_wavs = target_batch_wavs
        self.pos_fraction = pos_fraction
        self.shuffle = shuffle

        # wav indices per patient
        self.pid2idx = defaultdict(list)
        for i, pid in enumerate(patient_ids):
            self.pid2idx[pid].append(i)

        # patient labels
        self.pid2label = {}
        for i, pid in enumerate(patient_ids):
            if pid not in self.pid2label:
                self.pid2label[pid] = int(labels[i])

        self.pos_pids = [p for p, y in self.pid2label.items() if y == 1]
        self.neg_pids = [p for p, y in self.pid2label.items() if y == 0]

        # ---- CRITICAL: fixed epoch length ----
        total_wavs = len(patient_ids)
        self._len = math.ceil(total_wavs / target_batch_wavs)

    def __len__(self):
        return self._len

    def __iter__(self):
        for _ in range(self._len):
            batch = []
            remaining = self.target_batch_wavs

            # sample patients WITH replacement
            n_pos = int(self.pos_fraction * remaining)
            n_neg = remaining - n_pos

            # heuristic: average wavs per patient
            avg_wavs = sum(len(v) for v in self.pid2idx.values()) / len(self.pid2idx)
            est_patients = max(1, int(remaining / avg_wavs))

            k_pos = max(1, int(est_patients * self.pos_fraction))
            k_neg = est_patients - k_pos

            pos_pids = random.choices(self.pos_pids, k=k_pos)
            neg_pids = random.choices(self.neg_pids, k=k_neg)

            batch_pids = pos_pids + neg_pids
            random.shuffle(batch_pids)

            for pid in batch_pids:
                if remaining <= 0:
                    break

                wavs = self.pid2idx[pid]
                take = min(len(wavs), remaining)
                batch.extend(random.sample(wavs, take))
                remaining -= take

            yield batch