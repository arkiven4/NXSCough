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

import utils
import commons
import audio_processing
from augmentation import (
    DataAugmentator,
    ISD_additive_noise,
    LnL_convolutive_noise,
)

class CoughDatasets(torch.utils.data.Dataset):
    def __init__(self, data_numpy, hparams, train=True, wav_stats_path=None):
        self.audiopaths_and_text = shuffle(data_numpy, random_state=20)
        self.train = getattr(hparams, "train", train)

        self.hop_length = getattr(hparams, "hop_length", None)
        self.max_wav_value = getattr(hparams, "max_wav_value", None)
        self.mean_std_norm = getattr(hparams, "mean_std_norm", False)
        self.sampling_rate = getattr(hparams, "sampling_rate", None)
        self.saming_length = getattr(hparams, "saming_length", None)
        self.desired_length = getattr(hparams, "desired_length", None)
        self.pad_types = getattr(hparams, "pad_types", None)
        self.feature_type = getattr(hparams, "feature_type", None)
        self.db_path = getattr(hparams, "db_path", None)

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
        self.rezize_size = tuple(getattr(hparams, "rezize_size", [224, 224]))

        self.nClasses = getattr(hparams, "many_class", None)
        self.cough_detection = getattr(hparams, "cough_detection", None)
        self.processor = None

        if self.augment_data:
            self.data_augmentator = DataAugmentator(
                None,
                augmentation_noises_labels_path="/run/media/fourier/Data1/Pras/Interspeech2025/RIRS_NOISES/data_augmentation_noises_speechs_labels.tsv",
                augmentation_rirs_directory=None,
                augmentation_rirs_labels_path="/run/media/fourier/Data1/Pras/Interspeech2025/RIRS_NOISES/data_augmentation_rirs_labels.tsv",
                augmentation_window_size_secs=5.5, augmentation_probability=[0.3, 0.15, 0.2, 0.4, 0.0])

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
                        ref=1.0,
                    ),
                    dtype=torch.float32,
                )
            elif hparams.feature_type == "spectogram":
                self.wav_transform = lambda wav: torch.tensor(
                    librosa.amplitude_to_db(
                        np.abs(
                            librosa.stft(
                                y=wav.numpy() if isinstance(wav, torch.Tensor) else wav,
                                n_fft=hparams.filter_length,
                                hop_length=hparams.hop_length
                            )
                        ),
                        ref=1.0,
                    ),
                    dtype=torch.float32
                )

        if self.mix_audio == True:
            self.probs = [1 / self.nClasses] * self.nClasses

        random.seed(1234)
        #self._filter()
        

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
        # WARN : ONLY FOR COUGH DETECTIOP
        if self.cough_detection == True:
            wavname, dse_id, gndr_id, spk_id = audiopath_and_text[0], audiopath_and_text[1], 0, 0 #, audiopath_and_text[2], audiopath_and_text[3]
            wav = self.get_audio(self.db_path + "/" + wavname, start_index=audiopath_and_text[2], end_index=audiopath_and_text[3])
        else:
            wavname, dse_id, gndr_id, spk_id = audiopath_and_text[0], audiopath_and_text[1], audiopath_and_text[2], audiopath_and_text[3]
            wav = self.get_audio(self.db_path + "/" + wavname)

        wav, dse_id = self.mix_audio_sample(wav, dse_id)
        
        if self.mean_std_norm:
            wav = (wav - self.wav_mean) / (self.wav_std + 1e-6)
        elif self.max_wav_value:
            max_val = torch.max(torch.abs(wav))
            wav = wav / max_val if max_val != 0 else wav

        if self.wav_transform != None:
            wav = self.wav_transform(wav)  # [80, 224]
            # delta = torch.tensor(librosa.feature.delta(wav.numpy()), dtype=torch.float32)
            # delta2 = torch.tensor(librosa.feature.delta(wav.numpy(), order=2), dtype=torch.float32)
            # wav = torch.cat([wav, delta, delta2], dim=0)

            if self.multimask_augment == True and self.train == True:
                wav = audio_processing.multi_mask_spectrogram(
                    wav, tau=int(wav.shape[1] * self.tau),
                    nu=int(wav.shape[0] * self.nu),
                    num_masks=self.num_masks)  # T, F
            
            # WARN : ONLY FOR COUGH DETECTIOPN AND FIXED INPUT SIZE
            if self.cough_detection:
                wav = F.interpolate(wav.unsqueeze(0).unsqueeze(0), size=(240, 240), mode="bilinear", align_corners=False).squeeze(0).squeeze(0)
        
        wav = wav.unsqueeze(0)
        if self.mae_training == True:
            wav = self.transform_train(wav.unsqueeze(0)).squeeze(0)

        return (wavname, wav, dse_id, int(spk_id), int(gndr_id))

    def get_audio(self, filename, start_index=0, end_index=-1):
        audio = utils.load_audio_sample(filename, self.sampling_rate, self.saming_length,
                                        self.desired_length, fade_samples_ratio=self.fade_samples_ratio,
                                        pad_types=self.pad_types)  # repeat zero
        
        ## WARN : ONLY FOR COUGH DETECTIOPN
        if self.pad_types != "synthesis" and self.cough_detection:
            audio = audio[:, start_index:] if end_index == -1 else audio[:, start_index:end_index]
            
        if self.augment_data and self.train:
            if random.uniform(0, 0.999) > 1 - self.augment_prob:
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

        return audio.unsqueeze(0)

    def mix_audio_sample(self, wav, dse_id):
        if not (self.mix_audio and self.train):
            eye = np.eye(self.nClasses)
            dse_id = torch.from_numpy(eye[dse_id].astype(np.float32)).unsqueeze(0)
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

        wav_rand = self.get_audio(os.path.join(self.db_path, sampled_row[0]))

        sound1 = wav.squeeze(0).numpy()
        sound2 = wav_rand.squeeze(0).numpy()
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
    def __init__(self, many_data=2, processor=None, sampling_rate=16000):
        self.processor = processor
        self.sampling_rate = sampling_rate
        self.many_data = many_data

    def __call__(self, batch):
        lengths = torch.tensor([x[1].shape[-1] for x in batch])
        lengths_sorted, idx = torch.sort(lengths, descending=True)

        max_len = lengths_sorted[0]
        bsz = len(batch)

        first_wav = batch[0][1]
        feature_dim = first_wav.shape[1] if first_wav.ndim == 3 else 1

        wav_names = []
        wav_padded = torch.zeros(bsz, feature_dim, max_len, dtype=first_wav.dtype)
        attention_masks = torch.zeros(bsz, max_len)

        dse_ids = torch.zeros(bsz, self.many_data, dtype=torch.float32)
        spk_ids = torch.zeros(bsz, dtype=torch.long)
        gndr_ids = torch.zeros(bsz, dtype=torch.long)

        for i, j in enumerate(idx):
            name, wav, dse, spk, gndr = batch[j]
            wav_names.append(name)
            wav_length = wav.shape[-1]

            wav_padded[i, :, :wav_length] = wav
            attention_masks[i, :wav_length] = 1

            dse_ids[i] = dse
            spk_ids[i] = spk
            gndr_ids[i] = gndr

        return wav_names, wav_padded, attention_masks, dse_ids, [spk_ids, gndr_ids]

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
        dse_ids = torch.stack([torch.tensor(x[2]).squeeze(0) for x in batch])
        spk_ids = torch.stack([torch.tensor(x[3]) for x in batch])
        gndr_ids = torch.stack([torch.tensor(x[4]) for x in batch])

        audio_inputs = self.processor.feature_extractor(
            wavs,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
            # padding="max_length",
            return_attention_mask=True,
            padding=True
        )
        wav_padded = audio_inputs["input_features"]           # [B, n_mels, T]
        attention_masks = audio_inputs["attention_mask"]   # [B, T]

        return wav_name, wav_padded, attention_masks, dse_ids, [spk_ids, gndr_ids]
