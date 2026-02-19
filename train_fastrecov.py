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
from typing import Dict, List, Tuple
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
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split, StratifiedGroupKFold
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from tqdm import tqdm
from sklearn.mixture import GaussianMixture

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
from sklearn.metrics import (
    confusion_matrix, accuracy_score,
    balanced_accuracy_score,
    roc_auc_score, roc_curve
)

torch.set_float32_matmul_precision("medium")
cmap = cm.get_cmap("viridis")

#######################################################################
# REUSABLE FUNCTIONS
#######################################################################

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


# def get_sample_probs(weights:np.array, tau:float) -> np.array:
#     '''
#     Gets sampling probability based on weights
#     '''
#     exp_i = np.exp(weights/tau)
#     return exp_i/np.sum(exp_i)

def get_sample_probs(weights: np.ndarray, tau: float) -> np.ndarray:
    if tau == 0:
        raise ValueError("tau=0 should use deterministic sorting")
    z = weights / tau
    z = z - np.max(z)  # stable softmax
    exp_z = np.exp(z)
    return exp_z / exp_z.sum()

def sample_folds(n_folds:int, weights:np.array, tau:float) -> Tuple[List,Dict]:
    '''
    Given weights of the sample, performs weighted sampling into n_folds taking temperature tau into account
    Bigger τ = more uniform, more random, less influence from weights.
    '''
    n_samples = len(weights)
    #Assuming 'best' is the best fold and 'worst' is the worst fold and 'rest' are in between folds
    fold_id = {"best":[],"rest":[],"worst":[]}
    p_i = get_sample_probs(weights,tau)
    if tau<0.0001:
        #tau = 0 case, deterministic
        ordering = np.argsort(weights)[::-1]
    elif tau>10000:
        #infinite tau case
        ordering = np.random.permutation(len(weights))
    else:
        ordering = np.random.choice(n_samples,size=n_samples,replace=False,p=p_i)
    fold_id["best"] = ordering[:int(n_samples/n_folds)]
    fold_id["rest"] = ordering[int(n_samples/n_folds):(n_folds-1)*int(n_samples/n_folds)]
    fold_id["worst"] = ordering[(n_folds-1)*int(n_samples/n_folds):]
    # Convert into train_test splits
    rest_ids = fold_id["rest"]
    #split into n-2 folds with two reserved for best and worst
    rest_ids = [rest_ids[i*int(np.ceil(len(rest_ids)/(n_folds-2))):(i+1)*int(np.ceil(len(rest_ids)/(n_folds-2)))] for i in range(n_folds-2)]
    folds = [fold_id["best"]] + rest_ids + [fold_id["worst"]]
    fold_splits = []
    for i in range(n_folds):
        temp = folds.copy()
        test_ids = np.random.permutation(temp.pop(i))
        train_ids = np.random.permutation(np.concatenate(temp))
        fold_splits.append((train_ids,test_ids))
    return fold_splits, fold_id

def sample_folds_grouped(n_folds: int, weights: np.ndarray,
    tau: float, participant_ids: np.ndarray) -> Tuple[List, Dict]:
    """
    Weighted fold sampling with temperature τ, ensuring that
    each participant appears in only ONE fold.
    """

    # --- unique participants ---
    unique_pids, pid_inverse = np.unique(participant_ids, return_inverse=True)
    n_participants = len(unique_pids)

    # --- aggregate participant weights (mean is safest) ---
    part_weights = np.zeros(n_participants)
    for i in range(n_participants):
        part_weights[i] = weights[pid_inverse == i].mean()

    # --- compute sampling probabilities ---
    p_i = get_sample_probs(part_weights, tau)

    if tau < 1e-4:
        ordering = np.argsort(part_weights)[::-1]
    elif tau > 1e4:
        ordering = np.random.permutation(n_participants)
    else:
        ordering = np.random.choice(
            n_participants,
            size=n_participants,
            replace=False,
            p=p_i
        )

    # --- split participants into best/rest/worst ---
    fold_id = {"best": [], "rest": [], "worst": []}

    fold_size = int(n_participants / n_folds)

    fold_id["best"] = ordering[:fold_size]
    fold_id["rest"] = ordering[fold_size:(n_folds - 1) * fold_size]
    fold_id["worst"] = ordering[(n_folds - 1) * fold_size:]

    # split rest into n_folds-2 parts
    rest_ids = fold_id["rest"]
    chunk = int(np.ceil(len(rest_ids) / (n_folds - 2)))

    rest_ids = [
        rest_ids[i * chunk:(i + 1) * chunk]
        for i in range(n_folds - 2)
    ]

    part_folds = [fold_id["best"]] + rest_ids + [fold_id["worst"]]

    # --- expand participant folds → sample index folds ---
    fold_splits = []

    for i in range(n_folds):
        temp = part_folds.copy()

        test_pids = temp.pop(i)
        train_pids = np.concatenate(temp)

        # map participants → sample indices
        test_mask = np.isin(pid_inverse, test_pids)
        train_mask = np.isin(pid_inverse, train_pids)

        test_ids = np.random.permutation(np.where(test_mask)[0])
        train_ids = np.random.permutation(np.where(train_mask)[0])

        fold_splits.append((train_ids, test_ids))

    return fold_splits, fold_id

# -------------------------------------------------
# tau Updater schedule
# -------------------------------------------------
def update_tau_variance(tau, memory, alpha=5.0, tau_min=1e-4, tau_max=10.0):
    """
    Variance-driven temperature schedule
    Decrease tau when memory variance is large (signal clear),
    increase tau when variance is small (need exploration).
    Larger alpha -> tau decreases faster when variance is high
    """
    var = np.var(memory)
    tau_new = tau * np.exp(-alpha * var)
    return float(np.clip(tau_new, tau_min, tau_max))

def update_tau_validation(tau, fold_metrics, beta=1.5, tau_min=1e-4, tau_max=10.0):
    """
    Validation-metric feedback schedule
    If fold std is high (too greedy) → increase tau  -> (need exploration).
    If std is low (can exploit) → decrease tau  -> (signal clear).
    """
    std = np.std(fold_metrics)
    tau_new = tau * np.exp(beta * std)
    return float(np.clip(tau_new, tau_min, tau_max))

# -------------------------------------------------
# Noisy Rate Adaptive
# -------------------------------------------------
def update_noisy_drop_variance(memory, drop_min=0.1, drop_max=0.9, scale=5.0):
    """
    Increase drop rate as memory variance increases.
    """
    var = np.var(memory)
    drop = drop_min + (drop_max - drop_min) * (1 - np.exp(-scale * var))
    return float(np.clip(drop, drop_min, drop_max))


def update_noisy_drop_gmm(memory, drop_min=0.1, drop_max=0.9):
    mem = memory.reshape(-1, 1)

    gmm = GaussianMixture(n_components=2, random_state=0)
    gmm.fit(mem)

    means = np.sort(gmm.means_.flatten())
    separation = abs(means[1] - means[0])

    # normalize separation to [0,1] scale assumption
    sep_scaled = np.clip(separation, 0, 1)

    drop = drop_min + (drop_max - drop_min) * sep_scaled
    return float(np.clip(drop, drop_min, drop_max))

def update_noisy_drop_validation(current_drop, prev_val_mean, current_val_mean,
                                 step=0.05, drop_min=0.05, drop_max=0.95):
    """
    Increase drop if validation improves,
    decrease if validation degrades.
    """
    if current_val_mean > prev_val_mean:
        new_drop = current_drop + step
    else:
        new_drop = current_drop - step

    return float(np.clip(new_drop, drop_min, drop_max))

# -------------------------------------------------
# GMM-based threshold between clean vs noisy clusters
# -------------------------------------------------
def estimate_noise_threshold_gmm(memory):
    """
    Fit 2-component Gaussian mixture and
    return intersection midpoint between means.
    """
    mem = memory.reshape(-1, 1)

    gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=0)
    gmm.fit(mem)

    means = gmm.means_.flatten()
    order = np.argsort(means)

    noisy_mean = means[order[0]]
    clean_mean = means[order[1]]

    # midpoint between clusters (robust + simple)
    threshold = float((noisy_mean + clean_mean) / 2.0)
    return threshold

#######################################################################
# MAIN SCRIPT
#######################################################################
def main(cli_args=None):
    parser = train.parse_args()
    args = parser.parse_args(cli_args)

    model_dir = os.path.join("./logs", args.model_name)
    os.makedirs(model_dir, exist_ok=True)

    config_path = args.config_path if args.init else os.path.join(model_dir, "config.json")
    hps = train.load_config(config_path, model_dir, args)
    hps.model.spk_dim = 0
    # =============================================================
    # SECTION: Loading Data
    # =============================================================
    df_train, _ = train.load_data(hps)
    collate_fn = train.get_collate_fn(hps)
    target_labels = df_train[hps.data.target_column]
    print(target_labels.shape)

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
    logger.info(f"======================================")
    
    pool_net = train.setup_model(hps, is_init=args.init)
    # =============================================================
    # SECTION: Setup Logger, Dataloader
    # =============================================================
    N_RUNS = 100
    N_FOLDS = 7 # beta ↓ when N_FOLDS ↑
    RANDOM_STATE = 1
    TAU = 2

    NOISY_DROP = 0.1 # Dropping bottom N * NOISY_DROP of the dataset 
    WEIGHT_OBJ_FUNC = [1.0, 0.0, 0.0]
    WEIGHT_YOUDEN = [1.3, 0.7]

    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, random_state=RANDOM_STATE, shuffle=True)
    fold_splits = list(sgkf.split(X=df_train, y=target_labels, groups=df_train["participant"]))
    #kf = KFold(n_splits=N_FOLDS, random_state=RANDOM_STATE, shuffle=True)
    #fold_splits = list(kf.split(df_train, target_labels))

    runs_metadata = {
        'N_RUNS': N_RUNS,
        'N_FOLDS': N_FOLDS,
        'WEIGHT_OBJ_FUNC': WEIGHT_OBJ_FUNC,
        'WEIGHT_YOUDEN': WEIGHT_YOUDEN,
        'runs': {}
    }
    
    identified = []
    memory = np.zeros_like(target_labels, dtype=np.float64)
    print(memory.shape)
    for run_idx, seed in enumerate(range(RANDOM_STATE, RANDOM_STATE + N_RUNS)):
        logger.info(f"\n{'='*20} Run {seed}/{N_RUNS} {'='*20}")
        set_seed(seed)
        
        test_metadata = {
            "labels": [],
            "probs": [],
            "patient_ids": [],
            "ids": [],
        }
        for fold, (train_idx, test_idx) in enumerate(fold_splits):
            if len(identified) > 0:
                #NOISY_DROP = update_noisy_drop_variance(memory)
                NOISY_DROP = update_noisy_drop_gmm(memory)
                drop_idx = np.random.permutation(identified)[:int(NOISY_DROP * len(identified))]
                train_idx = list(set(train_idx) - set(drop_idx))

            trainval_fold = df_train.iloc[train_idx].reset_index(drop=True)
            test_fold = df_train.iloc[test_idx].reset_index(drop=True)

            mask_proper, mask_cp = train.stratified_group_holdout(trainval_fold[hps.data.target_column].values, trainval_fold["participant"].values, 
                                                                    test_size=0.15, seed=seed)
            
            train_fold = trainval_fold.iloc[mask_proper].reset_index(drop=True)
            val_fold = trainval_fold.iloc[mask_cp].reset_index(drop=True)

            train_loader, val_loader = train.prepare_fold_data(
                train_fold, val_fold, hps, collate_fn,
                use_precomputed=args.use_precomputed,
                precomputed_dir=args.precomputed_dir
            )

            pool_model = pool_net(**hps.model)
            runner_lightning = lightning_wrapper.CoughClassificationRunner(pool_model, hps=hps, custom_logger=logger, class_weights=[])

            checkpoint_callback = ModelCheckpoint(
                dirpath=f"{hps.model_dir}/fold_{fold}", monitor="val/loss",
                filename=f"pool_fold{fold}_{{epoch:02d}}", save_top_k=1, mode="min",
            )
            early_stopping = EarlyStopping(monitor="val/loss", patience=7, mode="min", verbose=False)
            trainer = L.Trainer(max_epochs=10000, callbacks=[checkpoint_callback, early_stopping],
                accelerator="gpu" if torch.cuda.is_available() else "cpu", devices="auto",
                default_root_dir=hps.model_dir
            )
            trainer.fit(runner_lightning, train_dataloaders=train_loader, val_dataloaders=val_loader)
            
            runner_lightning.calibrate_threshold = True
            trainer.test(runner_lightning, dataloaders=val_loader, ckpt_path="best")

            _, test_loader = train.prepare_fold_data(
                test_fold, test_fold, hps, collate_fn,
                use_precomputed=args.use_precomputed,
                precomputed_dir=args.precomputed_dir
            )
            trainer.test(runner_lightning, dataloaders=test_loader, ckpt_path="best")
            result_fold = runner_lightning.test_outputs

            test_metadata["labels"].append(result_fold['labels'])
            test_metadata["probs"].append(result_fold['probs'])
            test_metadata["patient_ids"].append(result_fold['patient_ids'])
            test_metadata["ids"].append(test_idx)
            
        test_labels_all = np.concatenate(test_metadata["labels"])
        pred_probs_all = np.concatenate(test_metadata["probs"])
        test_patient_ids_all = np.concatenate(test_metadata["patient_ids"])
        number_ids = [len(i) for i in test_metadata["ids"]]
        test_ids_all = np.concatenate(test_metadata["ids"])
        order = np.argsort(test_ids_all)

        # weights_acc = np.max(test_accs_np) - test_accs_np # distance from best fold, distance = 0, larger value.
        # weights_acc = safe_minmax_scale(weights_acc) # 0 - 1 Norm
        # weights_acc = 1 - weights_acc # Best fold → weight ≈ 1
        # weights_acc = np.concatenate([[weights_acc[i]]*number_ids[i] for i in range(N_FOLDS)])
        
        # “How confident is the model that the ground-truth label is correct?”
        # idx_temp = np.stack((np.arange(len(test_labels_all)), test_labels_all))
        # temp =  pred_probs_all[idx_temp[0,:], idx_temp[1,:]] # For Multiclass, Extract Only True Probs
        temp = np.where(test_labels_all == 1, pred_probs_all, 1 - pred_probs_all)
        weights_probs = (temp - temp.min()) / (temp.max() - temp.min() + 1e-8)
        weights = weights_probs #+ Objective_Function_Weights[1] * weights_acc
        weights = weights[order]

        # # Participant Aggregatiion
        # test_patient_ids_all = test_patient_ids_all[order]
        # _, inverse = np.unique(test_patient_ids_all, return_inverse=True)
        # pid_sum = np.bincount(inverse, weights=weights)
        # pid_cnt = np.bincount(inverse)
        # pid_mean = pid_sum / pid_cnt
        # weights = pid_mean[inverse]

        memory = 0.3 * weights + 0.7 * memory

        TAU = update_tau_variance(TAU, memory) # 1) variance-driven
        #TAU = update_tau_validation(TAU, test_youden) # 2) validation-feedback (use Youden or b_acc)
        MEMORY_NOISE_THRES = estimate_noise_threshold_gmm(memory)

        #fold_splits, _ = sample_folds(N_FOLDS, memory, TAU)
        fold_splits, _ = sample_folds_grouped(N_FOLDS, memory, participant_ids=df_train["participant"].values, tau=TAU)
        identified = np.where(memory <= MEMORY_NOISE_THRES)[0]

        runs_metadata['runs'][seed] = {
            'metrics': {
                'confidence_mean': np.mean(memory[memory > MEMORY_NOISE_THRES]),
                'identified_ratio': len(identified) / len(df_train),
                'roc-auc': roc_auc_score(test_labels_all, pred_probs_all),
            },
            'TAU': TAU,
            'MEMORY_NOISE_THRES': MEMORY_NOISE_THRES,
            'memory': memory,
        }

        for metric in runs_metadata['runs'][seed]['metrics'].keys():
            values_mean = []
            for _, values in runs_metadata['runs'].items():
                values_mean.append(values['metrics'][metric])
            plt.figure()
            plt.plot(values_mean)
            plt.xlabel("Runs")
            plt.ylabel(metric)
            plt.title(f"{metric} across runs")
            plt.grid(True)
            plt.savefig(f"{hps.model_dir}/{metric}_across_runs.png", dpi=300, bbox_inches="tight")
            plt.close()

        np.save(f"{hps.model_dir}/memory.npy", memory)
        np.save(f"{hps.model_dir}/identified.npy", identified)
        with open(os.path.join(hps.model_dir, "runs_metadata.pkl"), "wb") as f:
            pickle.dump(runs_metadata, f)

        for d in Path(hps.model_dir).glob("fold_*"):
            if d.is_dir():
                shutil.rmtree(d)
    
if __name__ == "__main__":
    main()
