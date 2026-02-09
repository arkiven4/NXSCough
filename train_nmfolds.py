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
def parse_args():
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
    parser.add_argument("--use_precomputed", action="store_true", help="Use precomputed features")
    parser.add_argument("--precomputed_dir", type=str, default=None, help="Directory with precomputed features")
    return parser


def main(cli_args=None):
    parser = parse_args()
    args = parser.parse_args(cli_args)

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
    hps = train.load_config(config_path, model_dir, args)
    hps.model.spk_dim = 0

    # =============================================================
    # SECTION: Loading Data
    # =============================================================
    df_train, _ = train.load_data(hps)
    collate_fn = train.get_collate_fn(hps)
    target_labels = df_train[hps.data.target_column]

    if not args.use_precomputed:
        utils.compute_spectrogram_stats_from_dataset(
            df_train, 
            hps.data, 
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
    if not args.eval:
        logger.info(f"✨ Tensorboard: http://100.101.198.75:{port}/#scalars&_smoothingWeight=0")
    else:
        logger.info(f"✨ Running in EVAL mode")
    logger.info(f"======================================")

    pool_net, pool_model = train.setup_model(hps, is_init=args.init)
    # =============================================================
    # SECTION: Setup Logger, Dataloader
    # =============================================================  
    if not args.eval:
        splitter_outter, num_folds_outter = train.create_data_split(df_train, target_labels, use_kfold=hps.train.use_Kfold, n_splits=10)
        for fold_outter, (trainval_idx, test_idx) in enumerate(splitter_outter):
            logger.info(f"\n{'='*20} Outter Fold {fold_outter+1}/{num_folds_outter} {'='*20}")

            trainval_fold = df_train.iloc[trainval_idx].reset_index(drop=True)
            test_fold = df_train.iloc[test_idx].reset_index(drop=True)
            
            fold_metrics = []
            fold_checkpoints = []
            fold_thresholds = []
            splitter_inner, num_folds_inner = train.create_data_split(trainval_fold, trainval_fold[hps.data.target_column], use_kfold=hps.train.use_Kfold)
            for fold_inner, (train_idx, val_idx) in enumerate(splitter_inner):
                logger.info(f"\n{'='*20} Inner Fold {fold_inner+1}/{num_folds_inner} {'='*20}")

                train_fold = trainval_fold.iloc[train_idx].reset_index(drop=True)
                val_fold = trainval_fold.iloc[val_idx].reset_index(drop=True)

                train_loader, val_loader = train.prepare_fold_data(
                    train_fold, val_fold, hps, fold_inner, collate_fn,
                    use_precomputed=args.use_precomputed,
                    precomputed_dir=args.precomputed_dir
                )

                pool_model = pool_net(hps.model.feature_dim, **hps.model)
                checkpoint_callback = ModelCheckpoint(
                    dirpath=f"{hps.model_dir}/{fold_outter}/fold_{fold_inner}",
                    monitor="val/loss",
                    filename=f"pool_fold{fold_inner}_{{epoch:02d}}",
                    save_top_k=1,
                    mode="min",
                )

                tb_logger = TensorBoardLogger(save_dir=f"{hps.model_dir}/{fold_outter}", name=f"fold_{fold_inner}", sub_dir="train")
                early_stopping = EarlyStopping(monitor="val/loss", patience=7, mode="min", verbose=False)
                runner_lightning = lightning_wrapper.CoughClassificationRunner(pool_model, hps=hps, custom_logger=logger, class_weights=[]) # Bcs i use Sampler
                trainer = L.Trainer(
                    max_epochs=1000,
                    callbacks=[checkpoint_callback, early_stopping],
                    logger=tb_logger,
                    accelerator="gpu" if torch.cuda.is_available() else "cpu",
                    devices="auto",
                    default_root_dir=f"{hps.model_dir}/{fold_outter}"
                )

                trainer.fit(runner_lightning, train_dataloaders=train_loader, val_dataloaders=val_loader)
                runner_lightning.calibrate_threshold = True
                trainer.test(runner_lightning, dataloaders=val_loader)[0]
                runner_lightning.calibrate_threshold = False

                results = trainer.test(runner_lightning, dataloaders=val_loader, ckpt_path="best")
                if results:
                    current_bacc = results[0].get('test_bacc', 0.0)
                    logger.info(f"Fold {fold_inner+1} Balanced Accuracy: {current_bacc:.4f}")
                    fold_metrics.append(current_bacc)
                    fold_checkpoints.append(checkpoint_callback.best_model_path)
                    fold_thresholds.append(runner_lightning.probs_threshold)

            if not args.eval and hps.train.use_Kfold:
                best_fold_idx, production_path = train.select_best_fold(fold_metrics, fold_checkpoints, f"{hps.model_dir}/{fold_outter}", logger)
                train.save_fold_info(best_fold_idx, fold_metrics, fold_thresholds, f"{hps.model_dir}/{fold_outter}")
            elif not args.eval:
                best_fold_idx = 0
                best_model_path = fold_checkpoints[best_fold_idx]
                production_path = os.path.join(f"{hps.model_dir}/{fold_outter}", "best_model.ckpt")
                shutil.copy2(best_model_path, production_path)
                train.save_fold_info(best_fold_idx, fold_metrics, fold_thresholds, f"{hps.model_dir}/{fold_outter}")
            else:
                # In eval mode, load best_fold_idx from existing info_fold.pkl
                info_fold_data = train.load_fold_info(f"{hps.model_dir}/{fold_outter}")
                best_fold_idx = info_fold_data.get("best_fold_idx", 0)


            runner_lightning = lightning_wrapper.CoughClassificationRunner.load_from_checkpoint(
                os.path.join(f"{hps.model_dir}/{fold_outter}", "best_model.ckpt"),
                model=pool_model,
                hps=hps, custom_logger=logger
            )
            runner_lightning.eval()
            trainer = L.Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu", devices="auto")

            info_fold = train.load_fold_info(f"{hps.model_dir}/{fold_outter}")
            best_fold_idx = info_fold["best_fold_idx"]
            fold_metrics = info_fold["fold_metrics"]
            runner_lightning.probs_threshold = info_fold['best_threshold']

            runner_lightning.generate_figure = True
            results_dict = train.evaluate_on_dataset(runner_lightning, trainer, test_fold, hps, best_fold_idx, collate_fn,
                    use_precomputed=args.use_precomputed,
                    precomputed_dir=args.precomputed_dir)
            train.write_results_to_file("result_overall.txt", results_dict, f"{hps.model_dir}", {0: "Test " + str(fold_outter)})
            runner_lightning.generate_figure = False

            # =============================================================
            # SECTION: Cleaning
            # =============================================================
            train.cleanup_fold_directories(f"{hps.model_dir}/{fold_outter}", best_fold_idx)


if __name__ == "__main__":
    main()
