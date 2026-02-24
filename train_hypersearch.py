# Standard library imports
import argparse
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

# Local imports
import commons
import lightning_wrapper
import models
import utils
import train
from cough_datasets import (
    CoughDatasets,
    CoughDatasetsCollate,
    CoughDatasetsProcessorCollate,
    CoughDetectionRatioBatchSampler,
    CoughDiseaseBinaryBatchSampler,
    PatientBatchSampler, AutoPatientBatchSampler
)

torch.set_float32_matmul_precision("medium")
cmap = cm.get_cmap("viridis")

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
    collate_fn = train.get_collate_fn(hps)
    target_labels = df_train[hps.data.target_column]

    if not args.use_precomputed:
        utils.compute_spectrogram_stats_from_dataset(
            df_train, hps.data,
            pickle_path=f"{hps.model_dir}/wav_stats.pickle"
        )

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
        params = {
            "hidden_dim_classifier": trial.suggest_categorical("hidden_dim_classifier", [32, 64, 128, 192, 256, 384, 512]),
            "dropout": trial.suggest_float("dropout", 0.1, 0.7),

            "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128, 192, 256, 384, 512]),
            "lstmnum_layers": trial.suggest_int("lstmnum_layers", 1, 4),
            "att_head": trial.suggest_categorical("att_head", [1, 2, 4, 8]),
            # "att_head_fusion": trial.suggest_categorical("att_head_fusion", [1, 2, 4, 8]),
            # "fusion_type": trial.suggest_categorical("fusion_type", ["gating", "cross_attn", "film"]),
            
            # "resnet_type": trial.suggest_categorical("resnet_type", ["resnet18", "resnet34", "resnet50", "resnet101"]),
            # "num_layers_resnet": trial.suggest_int("num_layers_resnet", 1, 4),
        }
        return params

    def objective(trial):
        now_params = sample_params(trial)
        fold_scores = []
        test_metadata = {"labels": [], "probs": []}

        splitter_outter, num_folds_outter = train.create_data_split(df_train, target_labels, use_kfold=True, n_splits=10, random_state=RNG_SEED)
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
                use_precomputed=args.use_precomputed,
                precomputed_dir=args.precomputed_dir
            )
            
            pool_model = pool_net(feature_dim=hps.model.feature_dim, use_tabular=False, **now_params)
            checkpoint_callback = ModelCheckpoint(
                dirpath=f"{hps.model_dir}/{fold_outter}",
                monitor="val/loss",
                filename=f"pool_{{epoch:02d}}",
                save_top_k=1,
                mode="min",
            )
            early_stopping = EarlyStopping(monitor="val/loss", patience=7, mode="min", verbose=False)

            runner_lightning = lightning_wrapper.CoughClassificationRunner(pool_model, hps=hps, custom_logger=logger, class_weights=[])
            trainer = L.Trainer(max_epochs=10000, callbacks=[checkpoint_callback, early_stopping],
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                devices="auto", default_root_dir=f"{hps.model_dir}/{fold_outter}", deterministic=True)
            trainer.fit(runner_lightning, train_dataloaders=train_loader, val_dataloaders=val_loader)

            production_path = os.path.join(f"{hps.model_dir}/{fold_outter}", "best_model.ckpt")
            shutil.move(trainer.checkpoint_callback.best_model_path, production_path)

            trainer = L.Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu", devices="auto")
            runner_lightning = lightning_wrapper.CoughClassificationRunner.load_from_checkpoint(
                os.path.join(f"{hps.model_dir}/{fold_outter}", "best_model.ckpt"),
                model=pool_model, hps=hps, custom_logger=logger
            )
            runner_lightning.eval()
            
            runner_lightning.calibrate_threshold = True
            trainer.test(runner_lightning, dataloaders=val_loader)

            results_dict = train.evaluate_on_dataset(runner_lightning, trainer, test_fold, hps, collate_fn,
                                                    use_precomputed=args.use_precomputed,
                                                    precomputed_dir=args.precomputed_dir)[0]

            fold_scores.append(results_dict['metrics'].get('test_bacc', 0.0))
            if os.path.exists(production_path):
                os.remove(production_path)
        
        mean_acc = np.mean(fold_scores)
        std_acc = np.std(fold_scores)

        alpha = 0.0
        final_score = mean_acc - alpha * std_acc
        return final_score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=40) # 70 -> 5 fold, 35 -> 10 fold

    print("Best params:", study.best_trial.params)
    print("Best stability-aware score:", study.best_value)

    best_result = {
        "params": study.best_trial.params,
        "score": study.best_value,
    }
    with open(f"{hps.model_dir}/optuna_best.pkl", "wb") as f:
        pickle.dump(best_result, f)

if __name__ == "__main__":
    main()
