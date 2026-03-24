# Standard library imports
import argparse
import gc
import importlib.util
import inspect
import json
import math
import os
import pickle
import random
import shutil
import socket
import subprocess
import sys
import tempfile
import warnings
import glob
from itertools import product
from pathlib import Path

# Third-party imports
import librosa
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from matplotlib import cm
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split, StratifiedGroupKFold
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from tqdm import tqdm
from sklearn.isotonic import IsotonicRegression
import optuna
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score

# Local imports
import commons
import lightning_wrapper
import models
import utils
import train
from cough_datasets import (
    CoughDatasets,
    CoughDatasetsCollate,
    CoughDetectionRatioBatchSampler,
    CoughDiseaseBinaryBatchSampler,
    PatientBatchSampler, AutoPatientBatchSampler
)
from precompute_features import precompute_features

torch.set_float32_matmul_precision("medium")
cmap = cm.get_cmap("viridis")


def _is_cuda_oom_error(exc: Exception) -> bool:
    visited = set()
    cur = exc
    while cur is not None and id(cur) not in visited:
        visited.add(id(cur))
        if isinstance(cur, torch.cuda.OutOfMemoryError):
            return True
        msg = str(cur).lower()
        if "out of memory" in msg and "cuda" in msg:
            return True
        if "cublas_status_alloc_failed" in msg or "cudnn_status_alloc_failed" in msg:
            return True
        cur = cur.__cause__ or cur.__context__
    return False

#######################################################################
# MAIN SCRIPT
#######################################################################
def main(cli_args=None):
    parser = train.parse_args()
    args = parser.parse_args(cli_args)

    RNG_SEED = 42
    L.seed_everything(RNG_SEED, workers=True)

    model_dir = os.path.join("./logs", args.model_name)
    os.makedirs(model_dir, exist_ok=True)

    config_path = args.config_path if args.init else os.path.join(model_dir, "config.json")
    hps = train.load_config(config_path, model_dir, args)
    hps.model.spk_dim = 0
    pool_net = train.setup_model(hps, is_init=args.init)

    # =============================================================
    # SECTION: Loading Data
    # =============================================================
    df_train, _ = train.load_data(hps)
    df_train = df_train[df_train["disease_status"] != 2]
    collate_fn = train.get_collate_fn(hps)
    target_labels = df_train[hps.data.target_column]

    labels_temp = torch.tensor(target_labels.values)
    num_pos = (labels_temp == 1).sum()
    num_neg = (labels_temp == 0).sum()
    pos_weight = num_neg / (num_pos + 1e-8)

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
    logger.info(f"======================================")

    # =============================================================
    # SECTION: Setup Logger, Dataloader
    # =============================================================
    def sample_params(trial):
        model_params = {
            "hidden_dim_classifier": trial.suggest_categorical("hidden_dim_classifier", [8, 16, 32, 64, 128, 192, 256, 384, 512]),
            "dropout": trial.suggest_float("dropout", 0.1, 0.9),

            "hidden_size": trial.suggest_categorical("hidden_size", [8, 16, 32, 64, 128, 192, 256, 384, 512]),
            "lstmnum_layers": trial.suggest_int("lstmnum_layers", 1, 4),
            # "att_head": trial.suggest_categorical("att_head", [1, 2, 4, 8]),
            # "att_head_fusion": trial.suggest_categorical("att_head_fusion", [1, 2, 4, 8]),
            # "fusion_type": trial.suggest_categorical("fusion_type", ["gating", "cross_attn", "film"]),

            # "resnet_type": trial.suggest_categorical("resnet_type", ["resnet18", "resnet34"]), # , "resnet50", "resnet101"
            # "num_layers_resnet": trial.suggest_int("num_layers_resnet", 1, 4),
        }

        #multimask_augment = trial.suggest_categorical("multimask_augment", [True, False])

        # filter_length = trial.suggest_categorical("filter_length", [512, 1024, 2048])
        # win_length = trial.suggest_categorical("win_length", [256, 512, 1024])
        # win_length = min(win_length, filter_length)  # ensure win_length <= filter_length

        data_params = {
            # "multimask_augment": multimask_augment,
            # "multimask_prob": trial.suggest_float("multimask_prob", 0.1, 0.8) if multimask_augment else 0.4,
            # "tau": trial.suggest_float("tau", 0.01, 0.5) if multimask_augment else 0.10,
            # "nu": trial.suggest_float("nu", 0.01, 0.5) if multimask_augment else 0.10,
            # "num_masks": trial.suggest_int("num_masks", 1, 6) if multimask_augment else 3,

            # "filter_length": filter_length,
            # "hop_length": trial.suggest_categorical("hop_length", [32, 64, 128, 256, 512]),
            # "win_length": win_length,
            # "n_mel_channels": trial.suggest_categorical("n_mel_channels", [40, 64, 80, 128, 160]), # [6, 13, 13 * 2, 13 * 3, 13 * 4] [40, 64, 80, 128, 160]
            # "mel_fmin": trial.suggest_categorical("mel_fmin", [0, 20, 40, 80, 100, 120, 200, 400, 500, 600]),
            # "mel_fmax": trial.suggest_categorical("mel_fmax", [5000, 6000, 7000, 8000]),

            # "delta_feature": trial.suggest_categorical("delta_feature", [True, False]),
            # "deltadelta_feature": trial.suggest_categorical("deltadelta_feature", [True, False]),

            # "augment_data": trial.suggest_categorical("augment_data", [True, False]),
            # "augment_prob": trial.suggest_float("augment_prob", 0.1, 0.8),
            # "augment_rawboost": trial.suggest_categorical("augment_rawboost", [True, False]),
            # "add_noise": trial.suggest_categorical("add_noise", [True, False]),
        }
        return model_params, data_params

    def objective(trial):
        model_params, data_params = sample_params(trial)

        # hps.data.multimask_augment = data_params["multimask_augment"]
        # hps.data.multimask_prob = data_params["multimask_prob"]
        # hps.data.tau = data_params["tau"]
        # hps.data.nu = data_params["nu"]
        # hps.data.num_masks = data_params["num_masks"]

        # hps.data.filter_length = data_params["filter_length"]
        # hps.data.hop_length = data_params["hop_length"]
        # hps.data.win_length = data_params["win_length"]
        # hps.data.n_mel_channels = data_params["n_mel_channels"]
        # hps.data.mel_fmin = data_params["mel_fmin"]
        # hps.data.mel_fmax = data_params["mel_fmax"]

        # hps.data.delta_feature = data_params["delta_feature"]
        # hps.data.deltadelta_feature = data_params["deltadelta_feature"]

        # hps.data.augment_data = data_params["augment_data"]
        # hps.data.augment_prob = data_params["augment_prob"]
        # hps.data.augment_rawboost = data_params["augment_rawboost"]
        # hps.data.add_noise = data_params["add_noise"]

        # feat_mult = 1
        # if data_params["delta_feature"]:
        #     feat_mult += 1
        # if data_params["deltadelta_feature"]:
        #     feat_mult += 1
        # hps.model.feature_dim = data_params["n_mel_channels"] * feat_mult
        # hps.model.feature_dim = data_params["n_mel_channels"]

        # Precompute features for this trial's spectrogram config into a temp dir
        trial_precomputed_dir = tempfile.mkdtemp(prefix=f"trial{trial.number}_", dir=hps.model_dir)
        try:
            logger.info(f"Precomputing features for trial {trial.number} -> {trial_precomputed_dir}")
            precompute_features(df_train, hps.data, trial_precomputed_dir, split_name="train")

            fold_scores = []
            test_metadata = {"labels": [], "probs": []}

            splitter_outter, num_folds_outter = train.create_data_split(df_train, target_labels, use_kfold=True, n_splits=hps.train.n_Kfold, random_state=RNG_SEED)
            for fold_outter, (inner_idx, test_idx) in enumerate(splitter_outter):
                logger.info(f"\n{'='*20} Outter Fold {fold_outter+1}/{num_folds_outter} {'='*20}")

                inner_fold = df_train.iloc[inner_idx].reset_index(drop=True)
                test_fold = df_train.iloc[test_idx].reset_index(drop=True)

                mask_proper, mask_cp = train.stratified_group_holdout(inner_fold[hps.data.target_column].values, inner_fold["participant"].values, 
                                                                      test_size=0.15, seed=RNG_SEED + fold_outter)

                train_fold = inner_fold.iloc[mask_proper].reset_index(drop=True)
                val_fold = inner_fold.iloc[mask_cp].reset_index(drop=True)
                
                train_loader, val_loader = train.prepare_fold_data(
                    train_fold, val_fold, hps, collate_fn,
                    use_precomputed=True,
                    precomputed_dir=trial_precomputed_dir
                )

                pool_model = pool_net(feature_dim=hps.model.feature_dim, use_tabular=False, **model_params)
                runner_lightning = lightning_wrapper.CoughClassificationRunner(pool_model, hps=hps, custom_logger=logger, class_weights=pos_weight)
                
                checkpoint_callback = ModelCheckpoint(
                    dirpath=f"{hps.model_dir}/fold_{fold_outter}",
                    monitor="val/loss",
                    filename=f"pool_{{epoch:02d}}",
                    save_top_k=1,
                    mode="min",
                )
                early_stopping = EarlyStopping(monitor="val/loss", patience=7, mode="min", verbose=False)
                trainer = L.Trainer(max_epochs=10000, callbacks=[checkpoint_callback, early_stopping],
                    accelerator="gpu" if torch.cuda.is_available() else "cpu",
                    devices="auto", default_root_dir=f"{hps.model_dir}/fold_{fold_outter}", deterministic=True)
                trainer.fit(runner_lightning, train_dataloaders=train_loader, val_dataloaders=val_loader)

                runner_lightning.calibrate_threshold = True
                trainer.test(runner_lightning, dataloaders=val_loader, ckpt_path="best")

                _, test_loader = train.prepare_fold_data(
                    test_fold, test_fold, hps, collate_fn,
                    use_precomputed=True,
                    precomputed_dir=trial_precomputed_dir
                )
                trainer.test(runner_lightning, dataloaders=test_loader, ckpt_path="best")
                result_fold = runner_lightning.test_outputs

                test_metadata["labels"].append(result_fold['labels'])
                test_metadata["probs"].append(result_fold['probs'])

                tn, fp, fn, tp = confusion_matrix(result_fold['labels'], result_fold['preds']).ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0   # Recall / TPR
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0   # TNR
                bacc = (sensitivity + specificity) / 2
                #fold_scores.append(bacc)
                fold_scores.append(roc_auc_score(result_fold['labels'], result_fold['probs']))

                del trainer, runner_lightning, pool_model, train_loader, val_loader, test_loader
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            test_labels_all = np.concatenate(test_metadata["labels"])
            pred_probs_all = np.concatenate(test_metadata["probs"])
            
            #final_score = -log_loss(test_labels_all, pred_probs_all)
            final_score = roc_auc_score(test_labels_all, pred_probs_all)

            # mean_acc = np.mean(fold_scores)
            # std_acc = np.std(fold_scores)
            # alpha = 0.0
            # final_score = mean_acc - alpha * std_acc
        except Exception as exc:
            if _is_cuda_oom_error(exc):
                logger.warning(f"Trial {trial.number} skipped due to CUDA OOM: {exc}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise optuna.exceptions.TrialPruned("Skipped trial due to CUDA OOM")
            raise
        finally:
            shutil.rmtree(trial_precomputed_dir, ignore_errors=True)
            for d in Path(hps.model_dir).glob("fold_*"):
                if d.is_dir():
                    shutil.rmtree(d)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return final_score

    def on_trial_end(study, trial):
        try:
            best_trial = study.best_trial
        except ValueError:
            return

        best_result = {
            "params": best_trial.params,
            "score": study.best_value,
            "study": study,
        }
        with open(f"{hps.model_dir}/optuna_best.pkl", "wb") as f:
            pickle.dump(best_result, f)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=200,
                   callbacks=[on_trial_end]) # 70 -> 5 fold, 35 -> 10 fold

    try:
        best_trial = study.best_trial
    except ValueError:
        print("No completed trials yet (all trials may have been pruned/failed).")
        return

    print("Best params:", best_trial.params)
    print("Best stability-aware score:", study.best_value)

    best_result = {
        "params": best_trial.params,
        "score": study.best_value,
        "study": study,
    }
    with open(f"{hps.model_dir}/optuna_best.pkl", "wb") as f:
        pickle.dump(best_result, f)

if __name__ == "__main__":
    main()
