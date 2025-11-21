import argparse, inspect, json, os, pickle, socket, subprocess, warnings, random, math, sys, importlib.util, shutil, tempfile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, auc, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score, roc_curve
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from tensorboard.backend.event_processing import event_accumulator
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoConfig, AutoFeatureExtractor, AutoModel, get_linear_schedule_with_warmup

import commons, models, utils
from cough_datasets import CoughDatasets, CoughDatasetsCollate
from wrapper.wav2vec import Wav2VecWrapper
from wrapper.wavlm_plus import WavLMWrapper
from wrapper.whisper import WhisperWrapper

import warnings
warnings.simplefilter("ignore", UserWarning)

def grl_lambda_schedule(current_step, max_steps, max_lambda=1.0):
    """Gradually increases λ from 0 to max_lambda."""
    if current_step < 1200:
        return 0.01
    p = (current_step - 1200) / (max_steps - 1200)
    p = min(max(p, 0.0), 1.0)
    return max_lambda * (2. / (1. + math.exp(-10 * p)) - 1.)

# =============================================================
# SECTION: Intialize Data
# =============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="try_wavlmlora_downstream")
args = parser.parse_args()

model_dir = os.path.join("./logs", args.model_name)
config_save_path = os.path.join(model_dir, "config.json")
with open(config_save_path, "r") as f:
    data = f.read()

config = json.loads(data)
  
hps = utils.HParams(**config)
hps.model_dir = model_dir

BATCH_SIZE = hps.train.batch_size
ACCUMULATION_STEP = hps.train.accumulation_steps
assert (ACCUMULATION_STEP > 0) and (BATCH_SIZE % ACCUMULATION_STEP == 0)
cur_bs = BATCH_SIZE // ACCUMULATION_STEP

# =============================================================
# SECTION: Loading Data
# =============================================================
df_train = pd.read_csv('/run/media/fourier/Data1/Pras/Thesis_Nexus/NXSCough/data/metadata.csv.train')
df_test = pd.read_csv('/run/media/fourier/Data1/Pras/Thesis_Nexus/NXSCough/data/metadata.csv.val')

df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

prob = pd.read_csv("/run/media/fourier/Data1/Pras/Thesis_Nexus/NXSCough/data/problematics.csv")
df_train = df_train[~df_train["path_file"].isin(prob["path_file"])]
df_test = df_test[~df_test["path_file"].isin(prob["path_file"])]

if hps.data.reorder_target:
    cols = hps.data.column_order
    df_train = df_train[cols]
    df_test = df_test[cols]

disease_codes = df_train[hps.data.target_column].unique().tolist()
class_frequencies = df_train[hps.data.target_column].value_counts().to_dict()
total_samples = len(df_train)
class_weights = {cls: total_samples / (len(disease_codes) * freq) if freq != 0 else 0 for cls, freq in class_frequencies.items()}
weights_list = [class_weights[cls] for cls in disease_codes]
class_weights_tensor = torch.tensor(weights_list, device='cuda', dtype=torch.float)
print(class_weights_tensor)
df_train = pd.read_csv('/run/media/fourier/Data1/Pras/Thesis_Nexus/NXSCough/data/metadata.csv.train')
df_test = pd.read_csv('/run/media/fourier/Data1/Pras/Thesis_Nexus/NXSCough/data/metadata.csv.val')

df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

prob = pd.read_csv("/run/media/fourier/Data1/Pras/Thesis_Nexus/NXSCough/data/problematics.csv")
df_train = df_train[~df_train["path_file"].isin(prob["path_file"])]
df_test = df_test[~df_test["path_file"].isin(prob["path_file"])]

if hps.data.reorder_target:
    cols = hps.data.column_order
    df_train = df_train[cols]
    df_test = df_test[cols]

disease_codes = df_train[hps.data.target_column].unique().tolist()
class_frequencies = df_train[hps.data.target_column].value_counts().to_dict()
total_samples = len(df_train)
class_weights = {cls: total_samples / (len(disease_codes) * freq) if freq != 0 else 0 for cls, freq in class_frequencies.items()}
weights_list = [class_weights[cls] for cls in disease_codes]
class_weights_tensor = torch.tensor(weights_list, device='cuda', dtype=torch.float)
print(class_weights_tensor)

pickle_path = 'wav_stats.pickle'
if not os.path.exists(pickle_path):
    means, stds = [], []
    paths = df_train['path_file'].dropna().tolist()

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

    with open(pickle_path, 'wb') as f:
        pickle.dump(stats, f)
# =============================================================
# SECTION: Setup Logger, Dataloader
# =============================================================
logger = utils.get_logger(hps.model_dir, filename='evaluate.log')
logger.info(hps)

collate_fn = CoughDatasetsCollate(hps.data.many_class)
train_dataset = CoughDatasets(df_train.values, hps.data, train=True)
val_dataset = CoughDatasets(df_test.values, hps.data, train=False)

train_loader = DataLoader(train_dataset, num_workers=28, shuffle=True, batch_size=cur_bs, pin_memory=True, drop_last=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, num_workers=28, shuffle=False, batch_size=hps.train.batch_size, pin_memory=True, drop_last=True, collate_fn=collate_fn)
# =============================================================
# SECTION: Setup Logger, Dataloader
# =============================================================
logger.info(f"Total Training Data : {len(df_train)}")
logger.info(f"======================================")
logger.info(f"✨ Training Input Shape: {next(iter(train_loader))[1][0].detach().cpu().numpy().shape}")
logger.info(f"✨ Loss: {hps.train.loss_function}")
logger.info(f"✨ Use Between Class Training: {hps.data.mix_audio}")
logger.info(f"✨ Use Augment: {hps.data.augment_data}")
logger.info(f"✨ Use Rawboost Augment: {hps.data.augment_rawboost}")
logger.info(f"✨ Padding Type: {hps.data.pad_types}")
logger.info(f"✨ Using Model: {hps.model.pooling_model}")
logger.info(f"======================================")

epoch_str = 1
global_step = 0
num_training_steps = len(train_loader) * 20
num_warmup_steps = int(0.01 * num_training_steps)  # 5% warmup

ssl_model_type = hps.model.ssl_model_type.lower()
ssl_model = None
if ssl_model_type == "wav2vec2":
    ssl_model = Wav2VecWrapper(hps.model)
    logger.info("✨ Using Wav2Vec2 SSL Model")
elif ssl_model_type == "wavlm":
    ssl_model = WavLMWrapper(hps.model)
    logger.info("✨ Using WavLM SSL Model")
elif ssl_model_type == "whisper":
    ssl_model = WhisperWrapper(hps.model)
    logger.info("✨ Using Whisper SSL Model")
elif ssl_model_type == "nonssl":
    logger.info("✨ Using Non-SSL Model")

if ssl_model != None:
    trainable_params = sum(p.numel() for p in ssl_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in ssl_model.parameters())
    trainable_percentage = 100 * trainable_params / total_params if total_params > 0 else 0
    logger.info(f'Trainable params: {trainable_params} | Total params: {total_params} | Trainable%: {trainable_percentage:.2f}% | Size: {trainable_params/(1e6):.2f}M')
    hps.model.feature_dim = ssl_model.hidden_size_ssl
hps.model.spk_dim = 0 #len(participant_mapping_longi)

temp_path = tempfile.NamedTemporaryFile(suffix=".py", delete=False).name
shutil.copy(f"{model_dir}/model_net.py.bak", temp_path)
spec = importlib.util.spec_from_file_location("model_net", temp_path)
model_net = importlib.util.module_from_spec(spec)
sys.modules["model_net"] = model_net
spec.loader.exec_module(model_net)

pool_net = getattr(model_net, hps.model.pooling_model)
pool_model = pool_net(hps.model.feature_dim, **hps.model)

if ssl_model != None:
    ssl_model.model_pooling = pool_model
    pool_model = ssl_model

pool_model = pool_model.cuda()

_, _, _, _, epoch_str = utils.load_checkpoint(
    os.path.join(hps.model_dir, "best_pool.pth"),
    pool_model,
    None,
    None,
)

# =============================================================
# Rerun Evaluate
# =============================================================
def evaluate_model(loader, split_name):
    pool_model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for _, (_, audio, attention_masks, dse_ids, _) in enumerate(tqdm(loader, desc=f"Eval {split_name}")):
            audio = audio.cuda(non_blocking=True).float().squeeze(1)
            attention_masks = attention_masks.cuda(non_blocking=True).float()
            dse_ids = dse_ids.cuda(non_blocking=True).float()

            logits = pool_model(audio, attention_mask=attention_masks)["disease_logits"]
            preds = torch.argmax(logits, dim=1)
            labels = np.argmax(dse_ids.cpu().numpy(), axis=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels)

    all_labels, all_preds = np.array(all_labels), np.array(all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    n_classes = cm.shape[0]
    class_labels = [f"Class {i+1}" for i in range(n_classes)]

    acc = accuracy_score(all_labels, all_preds)
    b_acc = balanced_accuracy_score(all_labels, all_preds)
    sens = np.mean([cm[i, i] / cm[i, :].sum() for i in range(n_classes) if cm[i, :].sum() > 0])
    spec = np.mean([(cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]) / (cm.sum() - cm[i, :].sum())
                    for i in range(n_classes) if (cm.sum() - cm[i, :].sum()) > 0])

    if split_name == "val":
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_labels, yticklabels=class_labels)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.savefig(f"{model_dir}/result_{split_name}_cm.png")

    return acc, b_acc, sens, spec

train_metrics = evaluate_model(train_loader, "train")
val_metrics = evaluate_model(val_loader, "val")

with open(f"{model_dir}/evaluate_summary.txt", "w") as f:
    f.write(
        f"==================================== Training Sets =====================================\n"
    )
    f.write(
        f"Train - Acc {train_metrics[0]:.2f} | BalAcc {train_metrics[1]:.2f} | "
        f"Sens {train_metrics[2]:.2f} | Spec {train_metrics[3]:.2f}\n"
    )
    f.write(
        f"Val - Acc {val_metrics[0]:.2f} | BalAcc {val_metrics[1]:.2f} | "
        f"Sens {val_metrics[2]:.2f} | Spec {val_metrics[3]:.2f}\n\n"
    )

# =============================================================
# TBCoda Solicited
# =============================================================
# df_solic = pd.read_csv('/run/media/fourier/Data1/Pras/DatabaseLLM/coda/solicited_original.csv')

# participant_mapping_longi = {participant: idx for idx, participant in enumerate(set(np.concatenate([df_solic['participant'].unique()])))} # df_solic['participant'].unique()
# df_solic['participant'] = df_solic['participant'].map(participant_mapping_longi)

# gender_mapping_longi = {gender: idx for idx, gender in enumerate(df_solic['sex'].unique())}
# df_solic['sex'] = df_solic['sex'].map(gender_mapping_longi)

# df_test = df_solic[hps.data.column_order]

# collate_fn = CoughDatasetsCollate(hps.data.many_class)
# val_dataset = CoughDatasets(df_test.values, hps.data, train=False)
# val_loader = DataLoader(val_dataset, num_workers=28, shuffle=False, batch_size=hps.train.batch_size, pin_memory=True, drop_last=True, collate_fn=collate_fn)

# cirdz_metrics = evaluate_model(val_loader, "coda_solicited")

# with open(f"{model_dir}/evaluate_summary.txt", "a") as f:
#     f.write(
#         f"==================================== TBCoda Solicited =====================================\n"
#     )
#     f.write(
#         f"Val - Acc {cirdz_metrics[0]:.2f} | BalAcc {cirdz_metrics[1]:.2f} | "
#         f"Sens {cirdz_metrics[2]:.2f} | Spec {cirdz_metrics[3]:.2f}\n\n"
#     )

# =============================================================
# CIRDZ
# =============================================================
hps.data.column_order = ["path_file", "disease_status", "sex", "participant"]
df = pd.read_csv('/run/media/fourier/Data1/Pras/DatabaseLLM/cirdz/metadata_wavs_filtered.csv')
participant_mapping_longi = {participant: idx for idx, participant in enumerate(set(np.concatenate([df['participant'].unique()])))}
df['participant'] = df['participant'].map(participant_mapping_longi)
gender_mapping_longi = {gender: idx for idx, gender in enumerate(df['sex'].unique())}
df['sex'] = df['sex'].map(gender_mapping_longi)

df_test = df[hps.data.column_order]

collate_fn = CoughDatasetsCollate(hps.data.many_class)
val_dataset = CoughDatasets(df_test.values, hps.data, train=False)
val_loader = DataLoader(val_dataset, num_workers=28, shuffle=False, batch_size=hps.train.batch_size, pin_memory=True, drop_last=True, collate_fn=collate_fn)

cirdz_metrics = evaluate_model(val_loader, "cirdz")

with open(f"{model_dir}/evaluate_summary.txt", "a") as f:
    f.write(
        f"==================================== CIRDZ =====================================\n"
    )
    f.write(
        f"Val - Acc {cirdz_metrics[0]:.2f} | BalAcc {cirdz_metrics[1]:.2f} | "
        f"Sens {cirdz_metrics[2]:.2f} | Spec {cirdz_metrics[3]:.2f}\n\n"
    )

# =============================================================
# TBScreen
# =============================================================
hps.data.column_order = ["path_file", "disease_status", "sex", "participant"]
df = pd.read_csv('/run/media/fourier/Data1/Pras/DatabaseLLM/TBscreen_Dataset/metadata_solicited.csv')
participant_mapping_longi = {participant: idx for idx, participant in enumerate(set(np.concatenate([df['participant'].unique()])))}
df['participant'] = df['participant'].map(participant_mapping_longi)
gender_mapping_longi = {gender: idx for idx, gender in enumerate(df['sex'].unique())}
df['sex'] = df['sex'].map(gender_mapping_longi)
df_test = df[hps.data.column_order]

collate_fn = CoughDatasetsCollate(hps.data.many_class)
val_dataset = CoughDatasets(df_test.values, hps.data, train=False)
val_loader = DataLoader(val_dataset, num_workers=28, shuffle=False, batch_size=hps.train.batch_size, pin_memory=True, drop_last=True, collate_fn=collate_fn)

cirdz_metrics = evaluate_model(val_loader, "tbscreen_solicited")

with open(f"{model_dir}/evaluate_summary.txt", "a") as f:
    f.write(
        f"==================================== TBScreen Solicited =====================================\n"
    )
    f.write(
        f"Val - Acc {cirdz_metrics[0]:.2f} | BalAcc {cirdz_metrics[1]:.2f} | "
        f"Sens {cirdz_metrics[2]:.2f} | Spec {cirdz_metrics[3]:.2f}\n\n"
    )

df = pd.read_csv('/run/media/fourier/Data1/Pras/DatabaseLLM/TBscreen_Dataset/metadata_longitudinal.csv')
participant_mapping_longi = {participant: idx for idx, participant in enumerate(set(np.concatenate([df['participant'].unique()])))}
df['participant'] = df['participant'].map(participant_mapping_longi)
gender_mapping_longi = {gender: idx for idx, gender in enumerate(df['sex'].unique())}
df['sex'] = df['sex'].map(gender_mapping_longi)
df_test = df[hps.data.column_order]

collate_fn = CoughDatasetsCollate(hps.data.many_class)
val_dataset = CoughDatasets(df_test.values, hps.data, train=False)
val_loader = DataLoader(val_dataset, num_workers=28, shuffle=False, batch_size=hps.train.batch_size, pin_memory=True, drop_last=True, collate_fn=collate_fn)

cirdz_metrics = evaluate_model(val_loader, "tbscreen_longitudinal")

with open(f"{model_dir}/evaluate_summary.txt", "a") as f:
    f.write(
        f"==================================== TBScreen Longitudinal =====================================\n"
    )
    f.write(
        f"Val - Acc {cirdz_metrics[0]:.2f} | BalAcc {cirdz_metrics[1]:.2f} | "
        f"Sens {cirdz_metrics[2]:.2f} | Spec {cirdz_metrics[3]:.2f}\n\n"
    )