import os, pickle, random
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchaudio import transforms as T
import torch.nn.functional as F
import torchvision.transforms as transforms

import librosa
from sklearn.utils import shuffle 

import utils, commons, audio_processing
from augmentation import DataAugmentator, ISD_additive_noise, LnL_convolutive_noise


class CoughDatasets(torch.utils.data.Dataset):
    def __init__(self, data_numpy, hparams, train=True):
        # ['FileName', 'EmoAct', 'EmoVal', 'EmoDom', 'SpkrID', 'EmoClassCode']
        self.audiopaths_and_text = shuffle(data_numpy, random_state=20)
        self.hop_length = hparams.hop_length
        self.max_wav_value = hparams.max_wav_value
        self.mean_std_norm = getattr(hparams, "mean_std_norm", False)
        self.sampling_rate = hparams.sampling_rate
        self.saming_length = hparams.saming_length
        self.desired_length = hparams.desired_length
        self.fade_samples_ratio = hparams.fade_samples_ratio
        self.pad_types = hparams.pad_types
        self.add_noise = hparams.add_noise
        self.db_path = hparams.db_path
        self.feature_type = hparams.feature_type

        self.train = train
        self.augment_data = hparams.augment_data
        self.augment_rawboost = hparams.augment_rawboost
        self.multimask_augment = hparams.multimask_augment
        self.tau = getattr(hparams, "tau", 0.0)
        self.nu = getattr(hparams, "nu", 0.0)
        self.num_masks = getattr(hparams, "num_masks", 0)
        self.mae_training = getattr(hparams, "mae_training", False)
        self.mix_audio = hparams.mix_audio
        self.nClasses = hparams.many_class
        print(self.train)

        if self.augment_data:
            print("Use Data Augmentation")
            self.data_augmentator = DataAugmentator(None, "/run/media/fourier/Data1/Pras/Interspeech2025/RIRS_NOISES/data_augmentation_noises_labels.tsv",
                None, "/run/media/fourier/Data1/Pras/Interspeech2025/RIRS_NOISES/data_augmentation_rirs_labels.tsv",
                5.5, ["apply_speed_perturbation", "apply_reverb", "add_background_noise", "apply_pitch_shift", "apply_random_gain"]) # "" apply_reverb add_background_noise

        if self.mean_std_norm:
            with open(f"wav_stats.pickle", 'rb') as f:
                stats = pickle.load(f)
                self.wav_mean, self.wav_std = stats["mean_db"], stats["std_db"]
                print(self.wav_mean)

        if self.mae_training:
            self.transform_train = transforms.Compose([
                            transforms.Resize((256, 256)),
                            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ])
            
        self.wav_transform = None
        if hparams.acoustic_feature:
            if hparams.feature_type == "mfcc":
                self.wav_transform = lambda wav: torch.tensor(
                    librosa.feature.mfcc(y=wav.numpy() if isinstance(wav, torch.Tensor) else wav, sr=self.sampling_rate, n_mfcc=13), dtype=torch.float32)
            elif hparams.feature_type == "melspectogram":
                self.wav_transform = commons.TacotronSTFT(
                                hparams.filter_length, hparams.hop_length, hparams.win_length,
                                hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
                                hparams.mel_fmax)
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
                        ref=np.max
                    ),
                    dtype=torch.float32
                )
        
        if self.mix_audio == True:
            print("Using BetweenClass Training All Same")
            self.probs = [1 / self.nClasses] * self.nClasses

        self._filter()
        #random.seed(1234)
        #random.shuffle(self.audiopaths_and_text)

    def _filter(self, min_sec=0.4):
        lengths = []
        new_audiopaths = []
        bytes_per_sample = 2  # for int16 PCM

        for audiopath_and_text in self.audiopaths_and_text:
            audiopath = os.path.join(self.db_path, audiopath_and_text[0])
            num_samples = os.path.getsize(audiopath) // bytes_per_sample
            duration_sec = num_samples / self.sampling_rate

            if duration_sec >= min_sec:
                lengths.append(num_samples // self.hop_length)
                new_audiopaths.append(audiopath_and_text)

        self.lengths = lengths
        self.audiopaths_and_text = np.array(new_audiopaths, dtype=object)

    def get_mel_text_pair(self, audiopath_and_text):
        wavname, dse_id, gndr_id, spk_id = audiopath_and_text[0], audiopath_and_text[1], audiopath_and_text[2], audiopath_and_text[3] #, audiopath_and_text[4] #, audiopath_and_text[2], audiopath_and_text[3], 0, audiopath_and_text[5] #audiopath_and_text[4], audiopath_and_text[5]
        wav = self.get_audio(self.db_path + "/" + wavname)

        if self.mix_audio == True and self.train == True:
            r = np.array(random.random())
            eye = np.eye(self.nClasses)
            while True:
                random_class = random.choices(range(self.nClasses), weights=self.probs, k=1)[0]
                if dse_id != random_class:
                    sampled_row = self.audiopaths_and_text[np.random.choice(np.where(self.audiopaths_and_text[:, 1] == random_class)[0])]
                    dse_id_rand = sampled_row[1]
                    dse_id = (eye[dse_id] * r + eye[dse_id_rand] * (1 - r)).astype(np.float32)
                    dse_id = torch.from_numpy(dse_id).unsqueeze(0)
                    break
                
            wav_rand = self.get_audio(self.db_path + "/" + sampled_row[0])
            
            sound1 = wav.squeeze(0).numpy()
            sound2 = wav_rand.squeeze(0).numpy()
            
            size = min(len(sound1), len(sound2)) 
            sound1 = sound1[:size]#wav[random.randint(0, len(wav) - size):][:size] if len(wav) > size else wav
            sound2 = sound2[:size]#wav_rand[random.randint(0, len(wav_rand) - size):][:size] if len(wav_rand) > size else wav_rand

            wav = audio_processing.mix(sound1, sound2, r, self.sampling_rate).astype(np.float32)
            wav = torch.from_numpy(wav)
        else:
            wav = wav.squeeze(0)
            eye = np.eye(self.nClasses)
            dse_id = (eye[dse_id]).astype(np.float32)
            dse_id = torch.from_numpy(dse_id).unsqueeze(0)
 
        if self.max_wav_value:
            max_val = torch.max(torch.abs(wav)) # Does It need again?
            wav = wav / max_val if max_val != 0 else wav
        if self.mean_std_norm:
            wav = (wav - self.wav_mean) / (self.wav_std + 1e-6)

        wav = wav.unsqueeze(0)

        if self.wav_transform != None:
            wav = self.wav_transform(wav.squeeze(0)) # [80, 224]
            #delta = torch.tensor(librosa.feature.delta(wav.numpy()), dtype=torch.float32)
            #delta2 = torch.tensor(librosa.feature.delta(wav.numpy(), order=2), dtype=torch.float32)
            #wav = torch.cat([wav, delta, delta2], dim=0)
            
            if self.multimask_augment == True and self.train == True:
                wav = audio_processing.multi_mask_spectrogram(wav, tau=int(wav.shape[1] * self.tau), nu=int(wav.shape[0] * self.nu), num_masks=self.num_masks) # T, F
            wav = wav.unsqueeze(0)

        if self.mae_training == True:
            wav = wav.unsqueeze(0)
            wav = self.transform_train(wav)
            wav = wav.squeeze(0)
            
        return (wavname, wav, dse_id, int(spk_id), int(gndr_id))

    def get_audio(self, filename): # random.randint(1, 6)
        audio = utils.load_audio_sample(filename, self.sampling_rate, self.saming_length, 
                                       self.desired_length, fade_samples_ratio=self.fade_samples_ratio, 
                                       pad_types=self.pad_types) # repeat zero
        audio = audio.squeeze(0)
        if self.augment_data and self.train:
            if random.uniform(0, 0.999) > 1 - 0.8:
                try:
                    audio = self.data_augmentator(audio.unsqueeze(0), self.sampling_rate).squeeze(0)
                except:
                    audio = audio

        original_feature_dtype = audio.dtype
        if self.augment_rawboost and self.train:        
            feature = LnL_convolutive_noise(audio.numpy(), 5, 5, 20, 8000, 100, 1000,
                                            10, 100, 0, 0, 5, 20, self.sampling_rate)
            feature = ISD_additive_noise(feature, 10, 2)
            if not isinstance(feature, torch.Tensor):
                feature = torch.tensor(feature)
            if feature.dtype != original_feature_dtype:
                feature = feature.to(original_feature_dtype)
            audio = feature

        if self.add_noise:
            audio = audio + torch.rand_like(audio)

        audio = audio.unsqueeze(0)
        return audio

    def __getitem__(self, index):
        return self.get_mel_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)

class CoughDatasetsCollate():
    """ Zero-pads model inputs and targets based on number of frames per step
    """
    def __init__(self, many_data=2):
        self.many_data = many_data

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        wav_name = []
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[1].shape[-1] for x in batch]),
            dim=0, descending=True)
        max_wav_len = input_lengths[0] # max([x[0].size(1) for x in batch])

        dse_ids = torch.FloatTensor(len(batch), self.many_data) # Change Depend on How Many Class
        spk_ids = torch.LongTensor(len(batch))
        gndr_ids = torch.LongTensor(len(batch))
    
        feature_dim = 1
        if len(batch[0][1].shape) == 3:
            feature_dim = batch[0][1].shape[1]

        wav_padded = torch.FloatTensor(len(batch), feature_dim, max_wav_len)
        wav_padded.zero_()

        attention_masks = torch.FloatTensor(len(batch), max_wav_len)
        attention_masks.zero_()

        for i in range(len(ids_sorted_decreasing)):
            wav_name.append(batch[i][0])
            wav = batch[ids_sorted_decreasing[i]][1]
            wav_padded[i, :, :wav.shape[-1]] = wav
            attention_masks[i, :wav.shape[-1]] = 1

            dse_ids[i, :] = batch[ids_sorted_decreasing[i]][2]
            spk_ids[i] = batch[ids_sorted_decreasing[i]][3]
            gndr_ids[i] = batch[ids_sorted_decreasing[i]][4]
        
        return wav_name, wav_padded, attention_masks, dse_ids, [spk_ids, gndr_ids]

############################ Multi Task Datasets

class MTCoughDatasets(torch.utils.data.Dataset):
    def __init__(self, data_numpy, hparams, train=True):
        # ['path_file', 'disease_label', 'smoker', 'hemoptysis', 'sex', 'age', 'tb_prior', 'tb_prior_Pul', 'tb_prior_Extrapul', 'tb_prior_Unknown', 'cough_score'],
        self.audiopaths_and_text = shuffle(data_numpy, random_state=20)
        self.hop_length = hparams.hop_length
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.desired_length = hparams.desired_length
        self.fade_samples_ratio = hparams.fade_samples_ratio
        self.pad_types = hparams.pad_types
        self.add_noise = hparams.add_noise
        self.db_path = hparams.db_path
        self.feature_type = hparams.feature_type

        self.train = train
        self.augment_data = hparams.augment_data
        self.multimask_augment = hparams.multimask_augment

        if self.augment_data:
            self.data_augmentator = DataAugmentator(None, "/run/media/fourier/Data1/Pras/Interspeech2025/RIRS_NOISES/data_augmentation_noises_labels.tsv",
                None, "/run/media/fourier/Data1/Pras/Interspeech2025/RIRS_NOISES/data_augmentation_rirs_labels.tsv",
                5.5, ["apply_speed_perturbation"])  # Using safer method

        self.wav_transform = None
        if hparams.acoustic_feature:
            if hparams.feature_type == "mfcc":
                self.wav_transform = T.MFCC(sample_rate=hparams.sampling_rate, n_mfcc=13, 
                                            melkwargs={"n_fft": hparams.win_length, "hop_length": hparams.hop_length, 
                                                       "n_mels": hparams.n_mel_channels, "window_fn": torch.hann_window,
                                                       "power": 2.0, "center": True, "normalized": False})
            elif hparams.feature_type == "melspectogram":
                self.wav_transform = commons.TacotronSTFT(
                                hparams.filter_length, hparams.hop_length, hparams.win_length,
                                hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
                                hparams.mel_fmax)
            elif hparams.feature_type == "spectogram":
                self.wav_transform = T.Spectrogram(
                    n_fft=hparams.win_length,
                    hop_length=hparams.hop_length,
                    power=2.0,
                    window_fn=torch.hann_window,
                    center=True,
                    normalized=False
                )

    def _filter(self):
        lengths = []
        for audiopath_and_text in self.audiopaths_and_text:
            audiopath = self.db_path + "/"  + audiopath_and_text[0]
            lengths.append(os.path.getsize(audiopath) // (2 * self.hop_length))
        self.lengths = lengths

    def get_mel_text_pair(self, audiopath_and_text):
        wavname, dse_id, speaker, gender, age = audiopath_and_text[0], audiopath_and_text[1], audiopath_and_text[2], audiopath_and_text[3], audiopath_and_text[4] #, audiopath_and_text[2], audiopath_and_text[3], 0, audiopath_and_text[5] #audiopath_and_text[4], audiopath_and_text[5]
        smoke_lweek, hemoptysis, cough_dur = audiopath_and_text[5], audiopath_and_text[6], audiopath_and_text[7]
        tb_type  = audiopath_and_text[8]
        
        wav = self.get_audio(self.db_path + "/" + wavname)

        max_val = torch.max(torch.abs(wav)) # Does It need again?
        wav = wav / max_val if max_val != 0 else wav
        wav = wav.unsqueeze(0)

        if self.wav_transform != None:
            wav = self.wav_transform(wav.squeeze(0)) # [80, 224]
            wav = (wav - wav.min()) / (wav.max() - wav.min() + 1e-8)
            print(wav.shape)

            if self.multimask_augment == True and self.train == True:
                wav = audio_processing.multi_mask_spectrogram(wav, tau=24, nu=15, num_masks=2)
            wav = wav.unsqueeze(0)

        return (wavname, wav, dse_id, [speaker, gender, smoke_lweek, hemoptysis, tb_type], [age, cough_dur])

    def get_audio(self, filename): # random.randint(1, 6)
        audio = utils.load_audio_sample(filename, self.sampling_rate, None, 
                                       self.desired_length, fade_samples_ratio=self.fade_samples_ratio, 
                                       pad_types=self.pad_types) # repeat zero
        audio = audio.squeeze(0)

        if self.augment_data:
            if random.uniform(0, 0.999) > 1 - 0.6:
                try:
                    audio = self.data_augmentator(audio.unsqueeze(0), self.sampling_rate).squeeze(0)
                    max_val = torch.max(torch.abs(audio))
                    audio = audio / max_val if max_val != 0 else audio
                except:
                    audio = audio

        if self.add_noise:
            audio = audio + torch.rand_like(audio)

        audio = audio.unsqueeze(0)
        return audio

    def __getitem__(self, index):
        return self.get_mel_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)

class MTCoughDatasetsCollate():
    """ Zero-pads model inputs and targets based on number of frames per step
    """
    def __init__(self, many_data=2):
        self.many_data = many_data

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        wav_name = []
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[1].shape[-1] for x in batch]),
            dim=0, descending=True)
        max_wav_len = input_lengths[0] # max([x[0].size(1) for x in batch])

        #wavname, wav, dse_id, [speaker, gender, smoke_lweek, hemoptysis, tb_type, age, cough_dur]
        dse_ids = torch.LongTensor(len(batch))

        spk_ids = torch.LongTensor(len(batch))
        gender_ids = torch.LongTensor(len(batch))
        smokers = torch.LongTensor(len(batch))
        hemoptysis = torch.LongTensor(len(batch))
        tb_types = torch.LongTensor(len(batch))

        ages = torch.FloatTensor(len(batch))
        cough_durs = torch.FloatTensor(len(batch))
    
        feature_dim = 1
        if len(batch[0][1].shape) == 3:
            feature_dim = batch[0][1].shape[1]

        wav_padded = torch.FloatTensor(len(batch), feature_dim, max_wav_len)
        wav_padded.zero_()

        attention_masks = torch.FloatTensor(len(batch), max_wav_len)
        attention_masks.zero_()

        for i in range(len(ids_sorted_decreasing)):
            wav_name.append(batch[i][0])
            wav = batch[ids_sorted_decreasing[i]][1]
            wav_padded[i, :, :wav.shape[-1]] = wav
            attention_masks[i, :wav.shape[-1]] = 1

            dse_ids[i] = batch[ids_sorted_decreasing[i]][2]

            spk_ids[i] = batch[ids_sorted_decreasing[i]][3][0]
            gender_ids[i] = batch[ids_sorted_decreasing[i]][3][1]
            smokers[i] = batch[ids_sorted_decreasing[i]][3][2]
            hemoptysis[i] = batch[ids_sorted_decreasing[i]][3][3]
            tb_types[i] = batch[ids_sorted_decreasing[i]][3][4]
            
            ages[i] = batch[ids_sorted_decreasing[i]][4][0]
            cough_durs[i] = batch[ids_sorted_decreasing[i]][4][1]
        
        return wav_name, wav_padded, attention_masks, dse_ids, [spk_ids, gender_ids, smokers, hemoptysis, tb_types], [ages, cough_durs]