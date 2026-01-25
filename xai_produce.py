import argparse, inspect, json, os, pickle, socket, subprocess, warnings, random, math, librosa, shutil
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning as L
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm

from functools import reduce
import umap
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import commons, models, utils, losses, lightning_wrapper
from cough_datasets import CoughDatasets, CoughDatasetsCollate, CoughDiseaseBinaryBatchSampler

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier


from sklearn.metrics import (
            confusion_matrix, accuracy_score,
            balanced_accuracy_score,
            roc_auc_score, roc_curve
        )


class ClassifierOutputSoftmaxTargetHola:
    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            return torch.softmax(model_output, dim=-1)[self.category]
        return torch.softmax(model_output, dim=-1)[:, self.category]
    
class CAMWrapper(torch.nn.Module):
    def __init__(self, lightning_model):
        super().__init__()
        self.lightning_model = lightning_model

    def forward(self, x):
        out = self.lightning_model(x)
        return out["disease_logits"]


def minmax_norm(x, eps=1e-8):
    return (x - x.min()) / (x.max() - x.min() + eps)

torch.set_float32_matmul_precision("medium")
cmap = cm.get_cmap("viridis")

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, ClassifierOutputSoftmaxTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


now_experiment = "resnet34_gtgram_deltadelta"
parser = argparse.ArgumentParser()
parser.add_argument("--init", action="store_true")
parser.add_argument("--model_name", type=str, default="try_wavlmlora_downstream")
parser.add_argument("--config_path", type=str, default="configs/ssl_finetuning.json")
args = parser.parse_args(["--model_name", now_experiment])

model_dir = os.path.join("./logs", args.model_name)
config_save_path = os.path.join(model_dir, "config.json")
with open(config_save_path, "r") as f:
    data = f.read()
config = json.loads(data)

hps = utils.HParams(**config)
hps.model_dir = model_dir
hps.data.mae_training = hps.train.mae_training
hps.data.ssccl_training = hps.train.ssccl_training
hps.model.spk_dim = 0

logger = utils.get_logger(hps.model_dir)
import sys, importlib.util, shutil, tempfile
temp_path = tempfile.NamedTemporaryFile(suffix=".py", delete=False).name
shutil.copy(f"{model_dir}/model_net.py.bak", temp_path)
spec = importlib.util.spec_from_file_location("model_net", temp_path)
model_net = importlib.util.module_from_spec(spec)
sys.modules["model_net"] = model_net
spec.loader.exec_module(model_net)
pool_net = getattr(model_net, hps.model.pooling_model)

hps.model.lora_finetune = False
pool_model = pool_net(hps.model.feature_dim, **hps.model)

runner_lightning = lightning_wrapper.CoughClassificationRunner(pool_model, hps=hps, custom_logger=logger, class_weights=[])
runner_lightning = lightning_wrapper.CoughClassificationRunner.load_from_checkpoint(
    os.path.join(f"{hps.model_dir}/best_model.ckpt"),
    model=pool_model,
    hps=hps, custom_logger=logger
)
runner_lightning.eval()

df_train = pd.read_csv(f'/run/media/fourier/Data1/Pras/Thesis_Nexus/NXSCough/data/{hps.data.metadata_csv}.train')
df_test = pd.read_csv(f'/run/media/fourier/Data1/Pras/Thesis_Nexus/NXSCough/data/{hps.data.metadata_csv}.test')

df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

df_train = df_train[hps.data.column_order]
df_test = df_test[hps.data.column_order]

collate_fn = CoughDatasetsCollate(hps.data.many_class)
target_labels = df_train[hps.data.target_column]

with open(os.path.join(hps.model_dir, "info_fold.pkl"), "rb") as f:
    info_fold = pickle.load(f)
    best_fold_idx = info_fold["best_fold_idx"]
    fold_metrics = info_fold["fold_metrics"]


val_dataset = CoughDatasets(df_test.values, hps.data,
                                wav_stats_path=f"{hps.model_dir}/wav_stats_fold_{best_fold_idx}.pickle", train=False)
val_loader = DataLoader(val_dataset, num_workers=28, shuffle=False,
                        batch_size=hps.train.batch_size, pin_memory=True, drop_last=False, collate_fn=collate_fn)


# Collect all predictions and labels from validation set
print("Collecting predictions from validation set...")
all_probs = []
all_preds = []
all_labels = []
all_audios = []
all_attention_masks = []
all_wavnames = []

runner_lightning.model.eval()
with torch.no_grad():
    for batch_data in tqdm(val_loader, desc="Processing batches"):
        wavnames, audio, _, attention_masks, dse_ids, [patient_ids, _, _] = batch_data
        audio = audio.cuda()
        attention_masks = attention_masks.cuda()
        
        out_model = runner_lightning.model.forward(audio, attention_mask=attention_masks)
        logits = out_model['disease_logits']
        probs = torch.sigmoid(logits).squeeze(-1)
        preds = (probs >= 0.5).long().cpu().numpy()
        labels = np.argmax(dse_ids.cpu().numpy(), axis=-1)
        
        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds)
        all_labels.extend(labels)
        all_audios.append(audio.cpu())
        all_attention_masks.append(attention_masks.cpu())
        all_wavnames.extend(wavnames)

all_probs = np.array(all_probs)
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_audios = torch.cat(all_audios, dim=0)
all_attention_masks = torch.cat(all_attention_masks, dim=0)

# Categorize predictions
tp_indices = np.where((all_preds == 1) & (all_labels == 1))[0]
tn_indices = np.where((all_preds == 0) & (all_labels == 0))[0]
fp_indices = np.where((all_preds == 1) & (all_labels == 0))[0]
fn_indices = np.where((all_preds == 0) & (all_labels == 1))[0]

print(f"\nTotal samples: {len(all_preds)}")
print(f"True Positives: {len(tp_indices)}")
print(f"True Negatives: {len(tn_indices)}")
print(f"False Positives: {len(fp_indices)}")
print(f"False Negatives: {len(fn_indices)}")

# Function to prepare mel spectrogram for visualization
def prepare_mel_rgb(audio_tensor):
    mel = audio_tensor.squeeze(0).cpu()
    # split
    mel_static = mel[0:80]
    mel_delta = mel[80:160]
    mel_deltadelta = mel[160:240]
    
    # normalize independently
    mel_static_n = minmax_norm(mel_static)
    mel_delta_n = minmax_norm(mel_delta)
    mel_deltadelta_n = minmax_norm(mel_deltadelta)
    
    # recombine
    mel_norm = np.vstack([mel_static_n, mel_delta_n, mel_deltadelta_n])
    mel_np = np.flipud(mel_norm)
    
    cmap = cm.get_cmap("viridis")
    mel_rgb = cmap(mel_np)[..., :3]
    return mel_rgb

# Function to generate CAM
def generate_cam(audio_tensor, target_class=0):
    cam_model = CAMWrapper(runner_lightning.model)
    target_layers = [runner_lightning.model.layer4, runner_lightning.model.layer3]
    targets = [ClassifierOutputTarget(target_class)]
    
    with GradCAM(model=cam_model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=audio_tensor, targets=targets)
        grayscale_cam = np.flipud(grayscale_cam[0, :])
    return grayscale_cam

# Create output directory
output_dir = os.path.join(hps.model_dir, "xai_analysis")
os.makedirs(output_dir, exist_ok=True)

# Select samples for each category (15 samples each)
n_samples = 15
categories = {
    'TP': tp_indices,
    'TN': tn_indices,
    'FP': fp_indices,
    'FN': fn_indices
}

# Generate individual CAMs and collect for mean calculation
for category_name, indices in categories.items():
    print(f"\nProcessing {category_name} samples...")
    
    if len(indices) == 0:
        print(f"No {category_name} samples found!")
        continue
    
    # Select random samples
    selected_indices = np.random.choice(indices, min(n_samples, len(indices)), replace=False)
    
    # Create category directory
    category_dir = os.path.join(output_dir, category_name)
    os.makedirs(category_dir, exist_ok=True)
    
    # Collect CAMs for mean calculation
    all_cams = []
    
    for idx, sample_idx in enumerate(tqdm(selected_indices, desc=f"Generating {category_name} CAMs")):
        audio_tensor = all_audios[sample_idx].cuda()
        mel_rgb = prepare_mel_rgb(all_audios[sample_idx])
        
        # Generate CAM
        grayscale_cam = generate_cam(audio_tensor)
        all_cams.append(grayscale_cam)
        
        # Create visualization
        visualization = show_cam_on_image(mel_rgb, grayscale_cam, use_rgb=True, image_weight=0.8)
        
        # Create figure with 3 panels
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Original mel spectrogram
        axes[0].imshow(mel_rgb)
        axes[0].set_title(f'Original Mel Spectrogram\n{category_name}', fontsize=14)
        axes[0].axis('off')
        
        # CAM heatmap
        axes[1].imshow(grayscale_cam, cmap='jet')
        axes[1].set_title(f'GradCAM Heatmap\nPred: {all_preds[sample_idx]}, GT: {all_labels[sample_idx]}, Prob: {all_probs[sample_idx]:.3f}', fontsize=14)
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(visualization)
        axes[2].set_title(f'Overlay\nFile: {all_wavnames[sample_idx]}', fontsize=14)
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(category_dir, f'{category_name}_{idx+1:02d}.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    # Generate mean CAM
    if len(all_cams) > 0:
        mean_cam = np.mean(all_cams, axis=0)
        
        # Create mean CAM visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Mean CAM heatmap
        im = axes[0].imshow(mean_cam, cmap='jet')
        axes[0].set_title(f'Mean GradCAM - {category_name}\n(n={len(all_cams)} samples)', fontsize=16, fontweight='bold')
        axes[0].axis('off')
        plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
        
        # Mean CAM with contours
        axes[1].imshow(mean_cam, cmap='jet')
        contours = axes[1].contour(mean_cam, levels=5, colors='white', linewidths=1.5, alpha=0.7)
        axes[1].clabel(contours, inline=True, fontsize=10)
        axes[1].set_title(f'Mean GradCAM with Contours - {category_name}', fontsize=16, fontweight='bold')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(category_dir, f'{category_name}_MEAN.png'), dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"Saved {len(all_cams)} individual CAMs and 1 mean CAM for {category_name}")

# Generate comparison plot of all mean CAMs
print("\nGenerating comparison plot of all mean CAMs...")
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
axes = axes.flatten()

for idx, (category_name, indices) in enumerate(categories.items()):
    if len(indices) == 0:
        axes[idx].text(0.5, 0.5, f'No {category_name} samples', 
                      ha='center', va='center', fontsize=16)
        axes[idx].axis('off')
        continue
    
    # Load or regenerate mean CAM
    selected_indices = np.random.choice(indices, min(n_samples, len(indices)), replace=False)
    all_cams = []
    
    for sample_idx in selected_indices[:n_samples]:
        audio_tensor = all_audios[sample_idx].cuda()
        grayscale_cam = generate_cam(audio_tensor)
        all_cams.append(grayscale_cam)
    
    mean_cam = np.mean(all_cams, axis=0)
    
    im = axes[idx].imshow(mean_cam, cmap='jet')
    axes[idx].set_title(f'{category_name} - Mean GradCAM (n={len(all_cams)})', 
                       fontsize=16, fontweight='bold')
    axes[idx].axis('off')
    plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'ALL_MEAN_CAMS_COMPARISON.png'), dpi=200, bbox_inches='tight')
plt.close()

print(f"\n✓ All visualizations saved to: {output_dir}")
print(f"  - Individual CAMs: {n_samples} per category")
print(f"  - Mean CAMs: 1 per category")
print(f"  - Comparison plot: ALL_MEAN_CAMS_COMPARISON.png")