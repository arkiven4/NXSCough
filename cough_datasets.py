import os, pickle, random
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchaudio import transforms as T
from sklearn.utils import shuffle 

import utils, commons, audio_processing
from augmentation import DataAugmentator

class AudioDatasetBaseFeature(torch.utils.data.Dataset):
    def __init__(self, data_path, df_data, sample_rate, desired_length):
        self.data = df_data
        self.sample_rate = sample_rate
        self.desired_length = desired_length
        self.wav_transform = T.MFCC(sample_rate=sample_rate, n_mfcc=13, melkwargs={ "n_fft": 2048,
                "hop_length": 512, "n_mels": 128, "window_fn": torch.hann_window,
                "power": 2.0, "center": True, "normalized": False})
        # self.wav_transform = T.Spectrogram(n_fft=1024, hop_length=64, power=2.0)
        # self.wav_transform = T.MelSpectrogram( sample_rate=sample_rate, n_fft=1024,
        #                         hop_length=256, n_mels=64, power=2.0)
        self.data = self.data.reset_index(drop=True)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        file_path, label = self.data.iloc[idx]
        audio_data = utils.load_audio_sample(file_path, self.sample_rate, None, 
                                       self.desired_length, fade_samples_ratio=6, pad_types='zero') # repeat zero

        audio_feat = self.wav_transform(audio_data)
        audio_feat = audio_feat.squeeze(0)

        # delta = torchaudio.functional.compute_deltas(audio_feat)
        # delta2 = torchaudio.functional.compute_deltas(delta)
        # audio_feat = torch.cat([audio_feat, delta, delta2], dim=0)

        return audio_feat.permute(1, 0), torch.tensor(label, dtype=torch.long)
    
def collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([seq.shape[1] for seq in sequences]) # [1, 31, 13]
    padded_sequences = nn.utils.rnn.pad_sequence(sequences, batch_first=True).squeeze(1)
    return padded_sequences, torch.tensor(labels), lengths

def df_sampler(train_df):
    class_counts = train_df['disease_label'].value_counts().sort_index().values
    total_samples = len(train_df)
    class_weights = torch.tensor(total_samples / (len(class_counts) * class_counts), dtype=torch.float32)

    df_counts = pd.DataFrame(train_df.groupby(['disease_label']).size().reset_index()).rename(columns={0:"counts"})
    df_counts['weights'] = df_counts['counts'].max() / df_counts['counts']
    df_balanced = pd.merge(train_df, df_counts[['disease_label','weights']], on='disease_label')
    sampler = torch.utils.data.WeightedRandomSampler(df_balanced['weights'].values, len(df_balanced))

    return sampler, class_weights

class CoughDatasets(torch.utils.data.Dataset):
    def __init__(self, data_numpy, hparams, train=True):
        # ['FileName', 'EmoAct', 'EmoVal', 'EmoDom', 'SpkrID', 'EmoClassCode']
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
        self.mix_audio = hparams.mix_audio
        self.nClasses = hparams.many_class
        print(self.train)

        if self.augment_data:
            self.data_augmentator = DataAugmentator(None, "/run/media/fourier/Data1/Pras/Interspeech2025/RIRS_NOISES/data_augmentation_noises_labels.tsv",
                None, "/run/media/fourier/Data1/Pras/Interspeech2025/RIRS_NOISES/data_augmentation_rirs_labels.tsv",
                5.5, ["apply_reverb", "add_background_noise"])

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
        
        if self.mix_audio == True:
            if False:
                print("Using BetweenClass Training With Weight")
                weights = [1.2742, 0.8229]
                inv_weights = [1 / w for w in weights]
                total = sum(inv_weights)
                self.probs = [w / total for w in inv_weights]
            else:
                print("Using BetweenClass Training All Same")
                self.probs = [1 / self.nClasses] * self.nClasses

        #self._filter()
        #random.seed(1234)
        #random.shuffle(self.audiopaths_and_text)

    def _filter(self):
        lengths = []
        for audiopath_and_text in self.audiopaths_and_text:
            audiopath = self.db_path + "/"  + audiopath_and_text[0]
            lengths.append(os.path.getsize(audiopath) // (2 * self.hop_length))
        self.lengths = lengths

    def get_mel_text_pair(self, audiopath_and_text):
        wavname, dse_id, spk_id = audiopath_and_text[0], audiopath_and_text[1], audiopath_and_text[4] #, audiopath_and_text[2], audiopath_and_text[3], 0, audiopath_and_text[5] #audiopath_and_text[4], audiopath_and_text[5]
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
            
            # desired_length = wav.shape[-1] / self.sampling_rate
            # wav_rand = audio_processing.cut_pad_sample_torchaudio(wav_rand, self.sampling_rate, desired_length, pad_types=self.pad_types)

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
        
        max_val = torch.max(torch.abs(wav)) # Does It need again?
        wav = wav / max_val if max_val != 0 else wav
        wav = wav.unsqueeze(0)

        if self.wav_transform != None:
            wav = self.wav_transform(wav.squeeze(0)) # [80, 224]
            if self.multimask_augment == True and self.train == True:
                wav = audio_processing.multi_mask_spectrogram(wav, tau=24, nu=15, num_masks=2)
            wav = wav.unsqueeze(0)

        return (wavname, wav, dse_id, spk_id)

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
        
        return wav_name, wav_padded, attention_masks, dse_ids, spk_ids
