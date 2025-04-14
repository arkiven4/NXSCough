import pickle
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchaudio import transforms as T

import utils

class AudioDataset(torch.utils.data.Dataset):
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

        with open(f"{data_path}/norm_stat.pkl", 'rb') as f:
            self.wav_mean, self.wav_std = pickle.load(f)
            print("Loaded Norm Stats")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        file_path, label = self.data.iloc[idx]
        audio_data = utils.load_audio_sample(file_path, self.sample_rate, [self.wav_mean, self.wav_std], 
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