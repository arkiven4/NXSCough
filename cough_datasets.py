from torch.utils.data import Sampler
from matplotlib import cm
import os
import pickle
import random

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

import librosa
import opensmile

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


class CoughDatasets(torch.utils.data.Dataset):
    def __init__(self, data_numpy, hparams, train=True, wav_stats_path=None):
        self.audiopaths_and_text = data_numpy
        self.train = getattr(hparams, "train", train)

        self.hop_length = getattr(hparams, "hop_length", None)
        self.max_wav_value = getattr(hparams, "max_wav_value", None)
        self.mean_std_norm = getattr(hparams, "mean_std_norm", False)
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

        if self.augment_data:
            # ["apply_speed_perturbation", "apply_pitch_shift", "apply_reverb", "add_background_noise", "apply_random_gain"]
            self.data_augmentator = DataAugmentator(
                None,
                augmentation_noises_labels_path="/run/media/fourier/Data1/Pras/Interspeech2025/RIRS_NOISES/data_augmentation_noises_speechs_labels.tsv",
                augmentation_rirs_directory=None,
                augmentation_rirs_labels_path="/run/media/fourier/Data1/Pras/Interspeech2025/RIRS_NOISES/data_augmentation_rirs_labels.tsv",
                augmentation_window_size_secs=5.5, augmentation_probability=[0.3, 0.25, 0.2, 0.25, 0.0])  # 0.3, 0.15, 0.2, 0.4, 0.0

        if self.mean_std_norm:
            with open(wav_stats_path, 'rb') as f:
                stats = pickle.load(f)
                self.wav_mean, self.wav_std = stats["mean_db"], stats["std_db"]

        if self.mae_training:
            self.transform_train = transforms.Compose([
                transforms.Resize(self.rezize_size),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        self.wav_transform = None

        if hparams.acoustic_feature:
            if hparams.feature_type == "mfcc":
                self.wav_transform = lambda wav: torch.tensor(librosa.feature.mfcc(
                    y=wav.numpy()
                    if isinstance(wav, torch.Tensor) else wav,
                    sr=self.sampling_rate, n_mfcc=13),
                    dtype=torch.float32)
            elif hparams.feature_type == "melspectogram":
                # self.wav_transform = commons.TacotronSTFT(
                #     hparams.filter_length, hparams.hop_length, hparams.win_length,
                #     hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
                #     hparams.mel_fmax)
                self.wav_transform = lambda wav: torch.tensor(
                    librosa.power_to_db(librosa.feature.melspectrogram(
                        y=wav.numpy() if isinstance(wav, torch.Tensor) else wav,
                        sr=hparams.sampling_rate,
                        n_mels=hparams.n_mel_channels,
                        fmin=hparams.mel_fmin,
                        fmax=hparams.mel_fmax,
                        n_fft=hparams.filter_length,
                        hop_length=hparams.hop_length,
                        power=2.0,
                    ), ref=np.max),
                    dtype=torch.float32,
                )
            elif hparams.feature_type == "teramel": 
                #(2048, 200, 800)
                #n_fft, hop_length, win_length
                self.wav_transform = lambda wav: torch.tensor(
                    np.log(librosa.feature.melspectrogram(
                        y=wav.numpy() if isinstance(wav, torch.Tensor) else wav,
                        sr=hparams.sampling_rate,
                        n_mels=hparams.n_mel_channels,
                        n_fft=hparams.filter_length,
                        hop_length=hparams.hop_length,
                        win_length=hparams.win_length,
                    ) + 1e-6),
                    dtype=torch.float32,
                )
            elif hparams.feature_type == "gammmaspectogram": 
                from gammatone.gtgram import gtgram
                self.wav_transform = lambda wav: torch.tensor(
                    np.log(gtgram(
                        wave=wav.numpy() if isinstance(wav, torch.Tensor) else wav,
                        fs=hparams.sampling_rate,
                        channels=hparams.n_mel_channels,
                        f_min=hparams.mel_fmin,
                        f_max=hparams.mel_fmax,
                        window_time=hparams.filter_length / hparams.sampling_rate,
                        hop_time=hparams.hop_length / hparams.sampling_rate, 
                    ) + 1e-8),
                    dtype=torch.float32,
                )
            elif hparams.feature_type == "spectogram": 
                self.wav_transform = lambda wav: torch.tensor(
                    librosa.power_to_db(
                        np.abs(
                            librosa.stft(
                                y=wav.numpy() if isinstance(wav, torch.Tensor) else wav,
                                n_fft=hparams.filter_length,
                                hop_length=hparams.hop_length
                            )
                        ) ** 2,
                        ref=np.max,
                    ),
                    dtype=torch.float32
                )
            elif hparams.feature_type == "opensmile":
                import joblib
                self.scaler = joblib.load("precomputed_stats/opensmile_global_scaler.pkl")
                self.wav_transform = lambda wav: torch.tensor(self.scaler.transform(
                    pd.concat(
                        [client.process_signal(wav, self.sampling_rate)
                         .reset_index(drop=True)
                         .add_prefix(f"{fs_name}__")
                         for fs_name, client in SMILE_CLIENTS.items()
                         ],
                        axis=1).fillna(0.0).values).T,
                    dtype=torch.float32
                )

            # WARN RESNET ONLY
            # from torchvision.transforms import (
            #     Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
            # )
            # self.img_transform = Compose([
            #     Resize([256], interpolation=InterpolationMode.BILINEAR),
            #     CenterCrop([224]),
            #     Normalize(mean=[0.485, 0.456, 0.406],
            #             std=[0.229, 0.224, 0.225]),
            # ])

        if self.mix_audio == True:
            self.probs = [1 / self.nClasses] * self.nClasses

        random.seed(1234)
        # self._filter()

    # def _filter(self, min_sec=0.4):
    #     lengths = []
    #     new_audiopaths = []
    #     bytes_per_sample = 2  # for int16 PCM

    #     for audiopath_and_text in self.audiopaths_and_text:
    #         audiopath = os.path.join(self.db_path, audiopath_and_text[0])
    #         num_samples = os.path.getsize(audiopath) // bytes_per_sample
    #         duration_sec = num_samples / self.sampling_rate

    #         if duration_sec >= min_sec:
    #             lengths.append(num_samples // self.hop_length)
    #             new_audiopaths.append(audiopath_and_text)

    #     self.lengths = lengths
    #     self.audiopaths_and_text = np.array(new_audiopaths, dtype=object)

    def get_mel_text_pair(self, audiopath_and_text):
        if self.cough_detection == True:
            # , audiopath_and_text[2], audiopath_and_text[3]
            wavname, dse_id, gndr_id, spk_id = audiopath_and_text[0], audiopath_and_text[1], 0, 0
            wav, dse_id = self.get_audio(
                self.db_path + "/" + wavname, dse_id, start_index=audiopath_and_text[2],
                end_index=audiopath_and_text[3])
            # print(wav.shape)
            wav2 = torch.empty(0, dtype=wav.dtype, device="cpu")
            return (wavname, wav, wav2, dse_id, int(spk_id), int(gndr_id))

        wavname, dse_id, gndr_id, spk_id = audiopath_and_text[0], audiopath_and_text[1], audiopath_and_text[2], audiopath_and_text[3]
        wav1, dse_id = self.get_audio(self.db_path + "/" + wavname, dse_id)

        if self.ssccl_training:
            wav2, _ = self.get_audio(self.db_path + "/" + wavname, always_augment=True)
        else:
            wav2 = torch.empty(0, dtype=wav1.dtype, device="cpu")

        if self.tabular_feature:
            tabular = torch.from_numpy(audiopath_and_text[4:8].astype("float32")).reshape(1, -1)
        else:
            tabular = np.zeros((1, 4))
            
        return (wavname, wav1, wav2, dse_id, int(spk_id), int(gndr_id), tabular)

    def get_audio(self, filename, dse_id=None, always_augment=False, start_index=0, end_index=-1):
        audio = utils.load_audio_sample(filename, self.sampling_rate, self.saming_length,
                                        self.desired_length, fade_samples_ratio=self.fade_samples_ratio,
                                        pad_types=self.pad_types)  # repeat zero
        audio = audio - audio.mean(dim=-1, keepdim=True)

        if self.pad_types != "synthesis" and self.cough_detection:
            if audio.shape[-1] < end_index:
                audio = audio[:, start_index:]
            else:
                audio = audio[:, start_index:end_index]
            # audio = audio[:, start_index:] if end_index == -1 else audio[:, start_index:end_index]
            if audio.shape[-1] == 0:
                print(audio.shape)

        if self.augment_data and self.train:
            if random.random() < self.augment_prob or always_augment:
                audio = self.data_augmentator(audio, self.sampling_rate)

        audio = audio.squeeze(0)
        if self.augment_rawboost and self.train:
            x = audio.numpy() if isinstance(audio, torch.Tensor) else audio
            x = LnL_convolutive_noise(
                x, 5, 5, 20, 8000, 100, 1000,
                10, 100, 0, 0, 5, 20, self.sampling_rate
            )
            x = ISD_additive_noise(x, 10, 2)
            audio = torch.as_tensor(x, dtype=audio.dtype)

        if self.add_noise:
            audio = audio + torch.rand_like(audio)

        audio = audio.unsqueeze(0)
        if dse_id != None:  # and self.train:
            audio, dse_id = self.mix_audio_sample(audio, dse_id)

        if self.wav_transform != None:
            if self.feature_type == "opensmile":
                audio = audio.numpy().reshape(-1)

            audio = self.wav_transform(audio)  # [80, 224]
            if self.delta_feature:
                audio_np = audio.detach().cpu().numpy()
                delta = torch.from_numpy(
                    librosa.feature.delta(audio_np)
                ).to(audio.device, dtype=audio.dtype)
                audio = torch.cat([audio, delta], dim=0)
                if self.deltadelta_feature:
                    delta2 = torch.from_numpy(
                        librosa.feature.delta(audio_np, order=2)
                    ).to(audio.device, dtype=audio.dtype)
                    audio = torch.cat([audio, delta2], dim=0)
                

            if self.multimask_augment == True and self.train == True:
                audio = audio_processing.multi_mask_spectrogram(
                    audio, tau=int(audio.shape[1] * self.tau),
                    nu=int(audio.shape[0] * self.nu),
                    num_masks=self.num_masks)  # T, F

            # if self.cough_detection:
            #     audio = F.interpolate(audio.unsqueeze(0).unsqueeze(0), size=(240, 240), mode="bilinear", align_corners=False).squeeze(0).squeeze(0)

            # if self.img_transform != None:
            #     audio = torch.flip(audio, dims=[0])
            #     mel_min = audio.min()
            #     mel_max = audio.max()
            #     audio = (audio - mel_min) / (mel_max - mel_min + 1e-6)
            #     idx = (audio * 255).clamp(0, 255).long()
            #     audio = viridis_lut[idx]
            #     wav = audio.permute(2, 0, 1).float()
            #     audio = self.img_transform(audio)

        if self.mean_std_norm:
            audio = (audio - self.wav_mean) / (self.wav_std + 1e-6)
        elif self.max_wav_value:
            max_val = torch.max(torch.abs(audio))
            audio = audio / max_val if max_val != 0 else audio

        audio = audio.unsqueeze(0)
        if self.mae_training == True:
            audio = self.transform_train(audio.unsqueeze(0)).squeeze(0)

        return audio, dse_id

    def mix_audio_sample(self, wav, dse_id):
        dse_id = int(dse_id)
        if not (self.mix_audio and self.train):
            if self.nClasses == 1:
                dse_id = torch.tensor([float(dse_id)], dtype=torch.float32)
            else:
                eye = torch.eye(self.nClasses, dtype=torch.float32)
                dse_id = eye[dse_id]
                dse_id = dse_id.unsqueeze(0)
            return wav.squeeze(0), dse_id

        r = np.array(random.random(), dtype=np.float32)
        eye = np.eye(self.nClasses)

        while True:
            random_class = random.choices(range(self.nClasses), weights=self.probs, k=1)[0]
            if dse_id != random_class:
                sampled_row = self.audiopaths_and_text[
                    np.random.choice(np.where(self.audiopaths_and_text[:, 1] == random_class)[0])
                ]
                dse_id_rand = sampled_row[1]
                dse_id = (eye[dse_id] * r + eye[dse_id_rand] * (1 - r)).astype(np.float32)
                dse_id = torch.from_numpy(dse_id).unsqueeze(0)
                break

        wav_rand, _ = self.get_audio(os.path.join(self.db_path, sampled_row[0]))

        sound1 = wav.squeeze(0).numpy()
        sound2 = wav_rand.squeeze(0, 1).numpy()
        size = min(len(sound1), len(sound2))

        sound1 = sound1[:size]
        sound2 = sound2[:size]

        mixed = audio_processing.mix(sound1, sound2, r, self.sampling_rate).astype(np.float32)
        wav = torch.from_numpy(mixed)

        return wav, dse_id

    def __getitem__(self, index):
        return self.get_mel_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class CoughDatasetsCollate:
    def __init__(self, many_data=2, many_tabular=4, processor=None, sampling_rate=16000):
        self.processor = processor
        self.sampling_rate = sampling_rate
        self.many_data = many_data
        self.many_tabular = many_tabular

    def __call__(self, batch):
        lengths1 = torch.tensor([x[1].shape[-1] for x in batch])
        lengths2 = torch.tensor([x[2].shape[-1] for x in batch])
        lengths_sorted1, idx = torch.sort(lengths1, descending=True)
        lengths_sorted2, idx = torch.sort(lengths2, descending=True)

        max_len = lengths_sorted1[0]
        bsz = len(batch)

        first_wav = batch[0][1]
        feature_dim = first_wav.shape[1] if first_wav.ndim == 3 else 1

        wav_names = []
        wav_padded1 = torch.zeros(bsz, feature_dim, lengths_sorted1[0], dtype=first_wav.dtype)
        wav_padded2 = torch.zeros(bsz, feature_dim, lengths_sorted2[0], dtype=first_wav.dtype)
        # wav_padded = torch.zeros(bsz, 3, 224, 224, dtype=first_wav.dtype)
        attention_masks = torch.zeros(bsz, max_len)

        dse_ids = torch.zeros(bsz, self.many_data, dtype=torch.float32)
        spk_ids = torch.zeros(bsz, dtype=torch.long)
        gndr_ids = torch.zeros(bsz, dtype=torch.long)
        tabular_ids = torch.zeros(bsz, self.many_tabular, dtype=torch.float32)

        for i, j in enumerate(idx):
            name, wav1, wav2, dse, spk, gndr, tblr = batch[j]
            wav_names.append(name)

            wav_padded1[i, :, :wav1.shape[-1]] = wav1
            wav_padded2[i, :, :wav2.shape[-1]] = wav2
            # wav_padded[i, :, :, :wav_length] = wav
            attention_masks[i, :wav1.shape[-1]] = 1

            dse_ids[i] = dse
            spk_ids[i] = spk
            gndr_ids[i] = gndr
            tabular_ids[i] = gndr

        return wav_names, wav_padded1, wav_padded2, attention_masks, dse_ids, [spk_ids, gndr_ids, tabular_ids]


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
        """
        wav_name = [x[0] for x in batch]
        wavs = [x[1].squeeze().numpy() for x in batch]  # list[np.ndarray]
        dse_ids = torch.stack([x[3].squeeze(0) for x in batch])
        spk_ids = torch.stack([torch.tensor(x[4]) for x in batch])
        gndr_ids = torch.stack([torch.tensor(x[5]) for x in batch])

        audio_inputs = self.processor.feature_extractor(
            wavs,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
            # padding="max_length",
            truncation = False, # ?
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

        return wav_name, wav_padded, dummy_wav, attention_masks, dse_ids, [spk_ids, gndr_ids]


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


from collections import defaultdict
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