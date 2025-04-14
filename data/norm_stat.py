import os
import numpy as np
import librosa
import pickle as pk

def get_norm_stat_for_wav(wav_list):
    """Compute mean and standard deviation of audio data."""
    count = 0
    wav_sum = 0
    wav_sqsum = 0

    for cur_wav in wav_list:
        wav_sum += np.sum(cur_wav)
        wav_sqsum += np.sum(cur_wav ** 2)
        count += len(cur_wav)

    wav_mean = wav_sum / count
    wav_var = (wav_sqsum / count) - (wav_mean ** 2)
    wav_std = np.sqrt(wav_var)

    return wav_mean, wav_std

def load_audio_files(folder):
    """Load all audio files from a given folder."""
    wav_list = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                audio, _ = librosa.load(file_path, sr=22050)  # Ensure consistent sampling rate
                audio = audio / 32768.0
                wav_list.append(audio)
    return wav_list

# Define folders
folders = ["TB_resample"]
 
# Collect all audio files
all_wavs = []
for folder in folders:
    if os.path.exists(folder):
        all_wavs.extend(load_audio_files(folder))
    else:
        print(f"Folder not found: {folder}")

# Compute statistics
print(len(all_wavs))
if all_wavs:
    wav_mean, wav_std = get_norm_stat_for_wav(all_wavs)

    # Save statistics
    with open("norm_stat.pkl", "wb") as f:
        pk.dump((wav_mean, wav_std), f)

    print(f"Normalization statistics saved: mean={wav_mean:.6f}, std={wav_std:.6f}")
else:
    print("No audio files found for normalization.")
