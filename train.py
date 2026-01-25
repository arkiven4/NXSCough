import argparse
import inspect
import json
import os
import pickle
import socket
import subprocess
import warnings
import random
import math
import librosa
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning as L
from torch.utils.data import WeightedRandomSampler
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

import commons
import models
import utils
import losses
import lightning_wrapper
from cough_datasets import CoughDatasets, CoughDatasetsCollate, CoughDatasetsProcessorCollate, CoughDiseaseBinaryBatchSampler, CoughDetectionRatioBatchSampler, PatientBatchSampler

torch.set_float32_matmul_precision("medium")
cmap = cm.get_cmap("viridis")

#######################################################################
parser = argparse.ArgumentParser()
parser.add_argument("--init", action="store_true")
parser.add_argument("--eval", action="store_true")
parser.add_argument("--model_name", type=str, default="try_wavlmlora_downstream")
parser.add_argument("--config_path", type=str, default="configs/ssl_finetuning.json")

parser.add_argument("--feature_dim", type=int)
parser.add_argument("--pooling_model", type=str)
parser.add_argument("--feature_type", type=str)
parser.add_argument("--delta_feature", action=argparse.BooleanOptionalAction, default=None)
parser.add_argument("--deltadelta_feature", action=argparse.BooleanOptionalAction, default=None)
parser.add_argument("--batch_size", type=int)

args = parser.parse_args()

model_dir = os.path.join("./logs", args.model_name)
os.makedirs(model_dir, exist_ok=True)

if not args.eval:
    port = utils.get_free_port()
    subprocess.Popen(
        ["tensorboard", "--logdir", model_dir, "--port", str(port), "--host", "0.0.0.0"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
else:
    port = None

config_path = args.config_path if args.init else os.path.join(model_dir, "config.json")
with open(config_path) as f:
    config = json.load(f)

# Centralized override map (single responsibility)
overrides = {
    ("model", "feature_dim"): args.feature_dim,
    ("model", "pooling_model"): args.pooling_model,
    ("data", "feature_type"): args.feature_type,
    ("data", "delta_feature"): args.delta_feature,
    ("data", "deltadelta_feature"): args.deltadelta_feature,
    ("train", "batch_size"): args.batch_size,
}

for (section, key), value in overrides.items():
    if value is not None:
        config.setdefault(section, {})[key] = value

with open(os.path.join(model_dir, "config.json"), "w") as f:
    json.dump(config, f, indent=2)

hps = utils.HParams(**config)
hps.model_dir = model_dir
hps.data.mae_training = hps.train.mae_training
hps.data.ssccl_training = hps.train.ssccl_training
# =============================================================
# SECTION: Loading Data
# =============================================================
df_train = pd.read_csv(f'/run/media/fourier/Data1/Pras/Thesis_Nexus/NXSCough/data/{hps.data.metadata_csv}.train')
df_test = pd.read_csv(f'/run/media/fourier/Data1/Pras/Thesis_Nexus/NXSCough/data/{hps.data.metadata_csv}.test')

# print(df_train['db'].value_counts())
# print(df_train['disease_status'].value_counts())
# #df_train = df_train[~df_train['db'].isin([0, 1])]

df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

df_train = df_train[hps.data.column_order]
df_test = df_test[hps.data.column_order]

if "qwen" in hps.model.pooling_model.lower():
    from transformers import Qwen3OmniMoeProcessor
    collate_fn = CoughDatasetsProcessorCollate(hps.data.many_class,
                                               processor=Qwen3OmniMoeProcessor.from_pretrained("/run/media/fourier/Data1/Pras/pretrain_models/Qwen3-Omni-30B-A3B-Thinking"),
                                               sampling_rate=hps.data.sampling_rate)
else:
    collate_fn = CoughDatasetsCollate(hps.data.many_class)
    
target_labels = df_train[hps.data.target_column]
# =============================================================
# SECTION: Model Setup
# =============================================================
logger = utils.get_logger(hps.model_dir)
logger.info(hps)

logger.info(f"======================================")
logger.info(f"✨ Loss: {hps.train.loss_function}")
logger.info(f"✨ Use Between Class Training: {hps.data.mix_audio}")
logger.info(f"✨ Use Augment: {hps.data.augment_data}")
logger.info(f"✨ Use Augment: Prob {hps.data.augment_prob}")
logger.info(f"✨ Use Rawboost Augment: {hps.data.augment_rawboost}")
logger.info(f"✨ Padding Type: {hps.data.pad_types}")
logger.info(f"✨ Using Model: {hps.model.pooling_model}")
if not args.eval:
    logger.info(f"✨ Tensorboard: http://100.101.198.75:{port}/#scalars&_smoothingWeight=0")
else:
    logger.info(f"✨ Running in EVAL mode")
logger.info(f"======================================")

hps.model.spk_dim = 0
if args.init:
    pool_net = getattr(models, hps.model.pooling_model)
    pool_model = pool_net(hps.model.feature_dim, **hps.model)
    shutil.copy2('./models.py', f'{hps.model_dir}/model_net.py.bak')
else:
    import sys
    import importlib.util
    import shutil
    import tempfile
    temp_path = tempfile.NamedTemporaryFile(suffix=".py", delete=False).name
    shutil.copy(f"{model_dir}/model_net.py.bak", temp_path)
    spec = importlib.util.spec_from_file_location("model_net", temp_path)
    model_modules = importlib.util.module_from_spec(spec)
    sys.modules["model_net"] = model_modules
    spec.loader.exec_module(model_modules)
    pool_net = getattr(model_modules, hps.model.pooling_model)
    pool_model = pool_net(hps.model.feature_dim, **hps.model)
# =============================================================
# SECTION: Setup Logger, Dataloader
# =============================================================
fold_metrics = []
fold_checkpoints = []

if not args.eval:
    if hps.train.use_Kfold:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        splitter = skf.split(df_train, target_labels)
        num_folds = skf.get_n_splits()
    else:
        train_idx, val_idx = train_test_split(
            df_train.index.to_numpy(),
            test_size=0.2,
            random_state=42,
            stratify=target_labels
        )
        splitter = [(train_idx, val_idx)]
        num_folds = 1

    for fold, (train_idx, val_idx) in enumerate(splitter):
        logger.info(f"\n{'='*20} Fold {fold+1}/{num_folds} {'='*20}")

        train_fold = df_train.iloc[train_idx].reset_index(drop=True)
        val_fold = df_train.iloc[val_idx].reset_index(drop=True)

        if hps.data.cough_detection:
            cough_idx = train_fold.index[train_fold["source"] == "cough"].tolist()
            speech_idx = train_fold.index[train_fold["source"] == "speech"].tolist()
            noise_idx = train_fold.index[train_fold["source"] == "noise"].tolist()

            sampler = CoughDetectionRatioBatchSampler(
                cough_idx=cough_idx,
                speech_idx=speech_idx,
                noise_idx=noise_idx,
                batch_size=hps.train.batch_size,
                ratios=(0.5, 0.35, 0.15)
            )
        else:
            positive_idx = train_fold.index[train_fold[hps.data.target_column] == 1].tolist()
            negative_idx = train_fold.index[train_fold[hps.data.target_column] == 0].tolist()

            num_pos = len(positive_idx)
            num_neg = len(negative_idx)

            # inverse-frequency class weights
            pos_weight = 1.0 / num_pos
            neg_weight = 1.0 / num_neg

            # initialize all weights to zero
            sample_weights = torch.zeros(len(train_fold), dtype=torch.double)

            # assign weights by index
            sample_weights[positive_idx] = pos_weight
            sample_weights[negative_idx] = neg_weight

            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            # sampler = CoughDiseaseBinaryBatchSampler(
            #     positive_idx,
            #     negative_idx,
            #     hps.train.batch_size,
            # )

            # patient_ids = train_fold['participant'].astype(str).values
            # sampler = PatientBatchSampler(
            #     patient_ids=patient_ids,
            #     patients_per_batch=hps.train.batch_size // 2,
            #     coughs_per_patient=2,
            # )

        class_weights_tensor = utils.compute_class_weights(train_fold, hps.data.target_column)
        if hps.data.acoustic_feature and hps.data.mean_std_norm == True:
            utils.compute_spectrogram_stats_from_dataset(
                train_fold, 
                hps.data, 
                pickle_path=f"{hps.model_dir}/wav_stats_fold_{fold}.pickle"
            )
        else:
            utils.compute_wav_stats(
                train_fold, 
                "path_file", 
                pickle_path=f"{hps.model_dir}/wav_stats_fold_{fold}.pickle"
            )

        train_dataset = CoughDatasets(train_fold.values, hps.data,
                                    wav_stats_path=f"{hps.model_dir}/wav_stats_fold_{fold}.pickle", train=True)
        val_dataset = CoughDatasets(val_fold.values, hps.data,
                                    wav_stats_path=f"{hps.model_dir}/wav_stats_fold_{fold}.pickle", train=False)

        train_loader = DataLoader(train_dataset, num_workers=28, sampler=sampler, batch_size=hps.train.batch_size,
                                  pin_memory=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, num_workers=28, shuffle=False, batch_size=hps.train.batch_size,
                                pin_memory=True, collate_fn=collate_fn)

        # Initialize a FRESH model for each fold
        hps.model.spk_dim = 0
        pool_model = pool_net(hps.model.feature_dim, **hps.model)

        checkpoint_callback = ModelCheckpoint(
            dirpath=f"{hps.model_dir}/fold_{fold}",
            monitor="val/loss",
            filename=f"pool_fold{fold}_{{epoch:02d}}",
            save_top_k=1,
            mode="min",
        )

        tb_logger = TensorBoardLogger(save_dir=hps.model_dir, name=f"fold_{fold}", sub_dir="train")
        early_stopping = EarlyStopping(monitor="val/loss", patience=7, mode="min", verbose=False)
        runner_lightning = lightning_wrapper.CoughClassificationRunner(
            pool_model, hps=hps, custom_logger=logger, class_weights=[]) # Bcs i use Sampler
        trainer = L.Trainer(
            max_epochs=1000,
            callbacks=[checkpoint_callback, early_stopping],
            logger=tb_logger,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices="auto",
            default_root_dir=hps.model_dir
        )
        trainer.fit(runner_lightning, train_dataloaders=train_loader, val_dataloaders=val_loader)
        results = trainer.test(runner_lightning, dataloaders=val_loader, ckpt_path="best")

        if results:
            current_bacc = results[0].get('test_bacc', 0.0)
            logger.info(f"Fold {fold+1} Balanced Accuracy: {current_bacc:.4f}")
            fold_metrics.append(current_bacc)
            fold_checkpoints.append(checkpoint_callback.best_model_path)

if not args.eval and hps.train.use_Kfold:
    mean_bacc = np.mean(fold_metrics)
    differences = [abs(metric - mean_bacc) for metric in fold_metrics]
    best_fold_idx = np.argmin(differences)
    best_fold_metric = fold_metrics[best_fold_idx]
    best_model_path = fold_checkpoints[best_fold_idx]

    logger.info(f"\n{'='*20} BEST MODEL SELECTION {'='*20}")
    logger.info(f"All Fold Balanced Accuracies: {[f'{m:.4f}' for m in fold_metrics]}")
    logger.info(f"Mean Balanced Accuracy: {mean_bacc:.4f}")
    logger.info(f"Best Fold (Closest to Mean): {best_fold_idx+1}")
    logger.info(f"Best Fold Balanced Accuracy: {best_fold_metric:.4f}")
    logger.info(f"Difference from Mean: {abs(best_fold_metric - mean_bacc):.4f}")
    logger.info(f"Source Checkpoint: {best_model_path}")

    if best_model_path and os.path.exists(best_model_path):
        production_path = os.path.join(hps.model_dir, "best_model.ckpt")
        shutil.copy2(best_model_path, production_path)
        logger.info(f"🏆 Saved Production Model to: {production_path}")
    else:
        logger.info("❌ Could not find best model checkpoint to copy.")

    payload = {
        "best_fold_idx": best_fold_idx,
        "fold_metrics": fold_metrics,
    }

    with open(os.path.join(hps.model_dir, "info_fold.pkl"), "wb") as f:
        pickle.dump(payload, f)
elif not args.eval:
    best_fold_idx = 0
    best_fold_metric = fold_metrics[best_fold_idx]
    best_model_path = fold_checkpoints[best_fold_idx]
    production_path = os.path.join(hps.model_dir, "best_model.ckpt")
    shutil.copy2(best_model_path, production_path)

    payload = {
        "best_fold_idx": best_fold_idx,
        "fold_metrics": fold_metrics,
    }

    with open(os.path.join(hps.model_dir, "info_fold.pkl"), "wb") as f:
        pickle.dump(payload, f)
else:
    # In eval mode, load best_fold_idx from existing info_fold.pkl
    if os.path.exists(os.path.join(hps.model_dir, "info_fold.pkl")):
        with open(os.path.join(hps.model_dir, "info_fold.pkl"), "rb") as f:
            info_fold_data = pickle.load(f)
            best_fold_idx = info_fold_data.get("best_fold_idx", 0)
    else:
        best_fold_idx = 0

# =============================================================
# SECTION: Test Phase
# =============================================================
db_map = {
    0: "TBCoda Logitudinal",
    1: "TBCoda Solicited",
    2: "TBScreen Logitudinal",
    3: "TBScreen Solicited",
    4: "CIRDZ",
    5: "UK19Covid",
}

runner_lightning = lightning_wrapper.CoughClassificationRunner(
    pool_model, hps=hps, custom_logger=logger, class_weights=[])
runner_lightning = lightning_wrapper.CoughClassificationRunner.load_from_checkpoint(
    os.path.join(hps.model_dir, "best_model.ckpt"),
    model=pool_model,
    hps=hps, custom_logger=logger
)
runner_lightning.eval()
trainer = L.Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu", devices="auto")

with open(os.path.join(hps.model_dir, "info_fold.pkl"), "rb") as f:
    info_fold = pickle.load(f)
best_fold_idx = info_fold["best_fold_idx"]
fold_metrics = info_fold["fold_metrics"]

# Handle result_summary.txt versioning
result_summary_path = f"{model_dir}/result_summary.txt"
if os.path.exists(result_summary_path):
    # Rotate existing files: old3 -> remove, old2 -> old3, old1 -> old2, current -> old1
    old3_path = f"{model_dir}/old3_result_summary.txt"
    old2_path = f"{model_dir}/old2_result_summary.txt"
    old1_path = f"{model_dir}/old1_result_summary.txt"
    
    if os.path.exists(old3_path):
        os.remove(old3_path)
    if os.path.exists(old2_path):
        shutil.move(old2_path, old3_path)
    if os.path.exists(old1_path):
        shutil.move(old1_path, old2_path)
    shutil.move(result_summary_path, old1_path)

with open(f"{model_dir}/result_summary.txt", "w") as f:
    f.write(f"=========================== Train Phase =============================\n")

df_test = pd.read_csv(f'/run/media/fourier/Data1/Pras/Thesis_Nexus/NXSCough/data/{hps.data.metadata_csv}.train')
df_test = df_test.reset_index(drop=True)
df_test = df_test[hps.data.column_order + ['db']]
for db_type in df_test['db'].unique().tolist():
    df_nowtest = df_test[df_test['db'] == db_type]
    val_dataset = CoughDatasets(df_nowtest.values, hps.data,
                                wav_stats_path=f"{hps.model_dir}/wav_stats_fold_{best_fold_idx}.pickle", train=False)
    val_loader = DataLoader(val_dataset, num_workers=28, shuffle=False,
                            batch_size=hps.train.batch_size, pin_memory=True, collate_fn=collate_fn)
    results = trainer.test(runner_lightning, dataloaders=val_loader)[0]
    with open(f"{model_dir}/result_summary.txt", "a") as f:
        f.write(
            f"{db_map[db_type]} - "
            f"Acc {results['test_acc']:.4f} | "
            f"BalAcc {results['test_bacc']:.4f} | "
            f"Sens {results['test_sens']:.4f} | "
            f"Spec {results['test_spec']:.4f} | "
            f"AUROC {results['test_auroc']:.4f} | "
            f"pAUROC {results['test_pauroc']:.4f}\n"
        )

with open(f"{model_dir}/result_summary.txt", "a") as f:
    f.write(f"\n=========================== Test Phase =============================\n")

df_test = pd.read_csv(f'/run/media/fourier/Data1/Pras/Thesis_Nexus/NXSCough/data/{hps.data.metadata_csv}.test')
df_test = df_test.reset_index(drop=True)
df_test = df_test[hps.data.column_order + ['db']]
for db_type in df_test['db'].unique().tolist():
    df_nowtest = df_test[df_test['db'] == db_type]
    val_dataset = CoughDatasets(df_nowtest.values, hps.data,
                                wav_stats_path=f"{hps.model_dir}/wav_stats_fold_{best_fold_idx}.pickle", train=False)
    val_loader = DataLoader(val_dataset, num_workers=28, shuffle=False,
                            batch_size=hps.train.batch_size, pin_memory=True, collate_fn=collate_fn)
    results = trainer.test(runner_lightning, dataloaders=val_loader)[0]
    with open(f"{model_dir}/result_summary.txt", "a") as f:
        f.write(
            f"{db_map[db_type]} - "
            f"Acc {results['test_acc']:.4f} | "
            f"BalAcc {results['test_bacc']:.4f} | "
            f"Sens {results['test_sens']:.4f} | "
            f"Spec {results['test_spec']:.4f} | "
            f"AUROC {results['test_auroc']:.4f} | "
            f"pAUROC {results['test_pauroc']:.4f}\n"
        )

df_test = pd.read_csv(f'/run/media/fourier/Data1/Pras/Thesis_Nexus/NXSCough/data/cirdz.csv.test')
df_test = df_test.reset_index(drop=True)
df_test = df_test[hps.data.column_order]
val_dataset = CoughDatasets(df_test.values, hps.data,
                            wav_stats_path=f"{hps.model_dir}/wav_stats_fold_{best_fold_idx}.pickle", train=False)
val_loader = DataLoader(val_dataset, num_workers=28, shuffle=False, batch_size=hps.train.batch_size,
                        pin_memory=True, collate_fn=collate_fn)
results = trainer.test(runner_lightning, dataloaders=val_loader)[0]
with open(f"{model_dir}/result_summary.txt", "a") as f:
    with open(f"{model_dir}/result_summary.txt", "a") as f:
        f.write(
            f"CIRDZ - "
            f"Acc {results['test_acc']:.4f} | "
            f"BalAcc {results['test_bacc']:.4f} | "
            f"Sens {results['test_sens']:.4f} | "
            f"Spec {results['test_spec']:.4f} | "
            f"AUROC {results['test_auroc']:.4f} | "
            f"pAUROC {results['test_pauroc']:.4f}\n"
        )

# =============================================================
# SECTION: Cleaning
# =============================================================
if os.path.isfile(os.path.join(hps.model_dir, "best_model.ckpt")):
    for name in os.listdir(hps.model_dir):
        p = os.path.join(hps.model_dir, name)
        if os.path.isdir(p) and name.startswith("fold_"):
            shutil.rmtree(p)
            print(f"Removed: {p}")
else:
    print("best_model.ckpt not found; no folders removed.")
