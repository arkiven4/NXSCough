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


def get_sample_probs(weights:np.array, tau:float) -> np.array:
    '''
    Gets sampling probability based on weights
    '''
    exp_i = np.exp(weights/tau)
    return exp_i/np.sum(exp_i)

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
# def get_sample_probs(weights: np.ndarray, tau: float) -> np.ndarray:
#     if tau == 0:
#         raise ValueError("tau=0 should use deterministic sorting")
#     z = weights / tau
#     z = z - np.max(z)  # stable softmax
#     exp_z = np.exp(z)
#     return exp_z / exp_z.sum()


# def sample_folds_grouped(
#     n_folds: int,
#     memory: np.ndarray,
#     participant_ids: np.ndarray,
#     tau: float,
#     random_state: int = None
# ) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], Dict]:

#     rng = np.random.default_rng(random_state)

#     # --------------------------------------------------
#     # 1. Aggregate memory at participant level
#     # --------------------------------------------------
#     unique_pids, inverse = np.unique(participant_ids, return_inverse=True)
#     pid_memory_sum = np.bincount(inverse, weights=memory)
#     pid_count = np.bincount(inverse)
#     pid_memory = pid_memory_sum / pid_count

#     n_groups = len(unique_pids)

#     # --------------------------------------------------
#     # 2. Temperature-weighted permutation of groups
#     # --------------------------------------------------
#     if tau == 0:
#         ordering = np.argsort(pid_memory)[::-1]
#     elif np.isinf(tau):
#         ordering = rng.permutation(n_groups)
#     else:
#         p = get_sample_probs(pid_memory, tau)
#         ordering = rng.choice(n_groups, size=n_groups, replace=False, p=p)

#     # --------------------------------------------------
#     # 3. Split groups evenly
#     # --------------------------------------------------
#     group_folds = np.array_split(ordering, n_folds)

#     fold_splits = []
#     fold_id = {}

#     for i in range(n_folds):
#         test_group_idx = group_folds[i]
#         train_group_idx = np.concatenate([g for j, g in enumerate(group_folds) if j != i])

#         test_pids = unique_pids[test_group_idx]
#         train_pids = unique_pids[train_group_idx]

#         test_mask = np.isin(participant_ids, test_pids)
#         train_mask = np.isin(participant_ids, train_pids)

#         test_ids = np.where(test_mask)[0]
#         train_ids = np.where(train_mask)[0]

#         fold_splits.append((train_ids, test_ids))

#     fold_id["group_memory"] = pid_memory
#     fold_id["group_order"] = ordering

#     return fold_splits, fold_id

# -------------------------------------------------
# Variance-driven temperature schedule
# -------------------------------------------------
def update_tau_variance(tau, memory, alpha=2.0, tau_min=1e-4, tau_max=10.0):
    """
    Decrease tau when memory variance is large (signal clear),
    increase tau when variance is small (need exploration).
    """
    var = np.var(memory)
    tau_new = tau * np.exp(-alpha * var)
    return float(np.clip(tau_new, tau_min, tau_max))

# -------------------------------------------------
# Validation-metric feedback schedule
# -------------------------------------------------
def update_tau_validation(tau, fold_metrics, beta=1.5,
                          tau_min=1e-4, tau_max=10.0):
    """
    If fold std is high → increase tau (too greedy).
    If std is low → decrease tau (can exploit).
    """
    std = np.std(fold_metrics)
    tau_new = tau * np.exp(beta * std)
    return float(np.clip(tau_new, tau_min, tau_max))

# -------------------------------------------------
# Noisy Rate Adaptive
# -------------------------------------------------
def update_noisy_drop_variance(memory,
                               drop_min=0.1,
                               drop_max=0.9,
                               scale=5.0):
    """
    Increase drop rate as memory variance increases.
    """
    var = np.var(memory)
    drop = drop_min + (drop_max - drop_min) * (1 - np.exp(-scale * var))
    return float(np.clip(drop, drop_min, drop_max))


def update_noisy_drop_gmm(memory,
                          drop_min=0.1,
                          drop_max=0.9):
    mem = memory.reshape(-1, 1)

    gmm = GaussianMixture(n_components=2, random_state=0)
    gmm.fit(mem)

    means = np.sort(gmm.means_.flatten())
    separation = abs(means[1] - means[0])

    # normalize separation to [0,1] scale assumption
    sep_scaled = np.clip(separation, 0, 1)

    drop = drop_min + (drop_max - drop_min) * sep_scaled
    return float(np.clip(drop, drop_min, drop_max))

def update_noisy_drop_validation(current_drop,
                                 prev_val_mean,
                                 current_val_mean,
                                 step=0.05,
                                 drop_min=0.05,
                                 drop_max=0.95):
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


def safe_minmax_scale(x, eps=1e-8):
    arr = np.asarray(x, dtype=np.float64)
    span = arr.max() - arr.min()
    if span < eps:
        return np.full_like(arr, 0.5, dtype=np.float64)
    return (arr - arr.min()) / (span + eps)


def select_identified_with_warmup(memory,
                                  threshold,
                                  run_idx,
                                  warmup_runs=10,
                                  max_ratio=0.10,
                                  cap_after_warmup=False):
    """
    Select noisy IDs with warmup cap.
    During warmup, cap grows linearly up to max_ratio.
    After warmup, cap is optional via cap_after_warmup.
    """
    candidate = np.where(memory <= threshold)[0]
    if len(candidate) == 0:
        return np.array([], dtype=np.int64), 0.0

    in_warmup = run_idx < warmup_runs
    if in_warmup:
        warmup_factor = min(1.0, (run_idx + 1) / max(1, warmup_runs))
        cap_ratio = max_ratio * warmup_factor
        cap_n = int(np.floor(cap_ratio * len(memory)))
    elif cap_after_warmup:
        cap_ratio = max_ratio
        cap_n = int(np.floor(cap_ratio * len(memory)))
    else:
        cap_ratio = len(candidate) / max(1, len(memory))
        cap_n = len(candidate)

    if cap_n <= 0:
        return np.array([], dtype=np.int64), cap_ratio

    candidate_sorted = candidate[np.argsort(memory[candidate])]  # ascending => worst first
    selected = candidate_sorted[:min(cap_n, len(candidate_sorted))]
    return selected.astype(np.int64), cap_ratio


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
    
    pool_net, pool_model = train.setup_model(hps, is_init=args.init)
    # =============================================================
    # SECTION: Setup Logger, Dataloader
    # =============================================================
    N_RUNS = 100
    N_FOLDS = 6 # beta ↓ when N_FOLDS ↑
    RANDOM_STATE = 1
    TAU = 2

    NOISY_DROP = 0.1 # Dropping bottom N * NOISY_DROP of the dataset 
    Objective_Function_Weights = [0.6, 0.15, 0.25]
    Youden_Weights = [1.3, 0.7]
    WARMUP_RUNS = max(10, int(0.1 * N_RUNS))
    MAX_IDENTIFIED_RATIO = 0.10
    CAP_AFTER_WARMUP = False

    kf = KFold(n_splits=N_FOLDS, random_state=RANDOM_STATE, shuffle=True)
    fold_splits = list(kf.split(df_train, target_labels))

    monitor_metrics = {
        'N_RUNS': N_RUNS,
        'N_FOLDS': N_FOLDS,
        'Objective_Function_Weights': Objective_Function_Weights,
        'Youden_Weights': Youden_Weights,
        'runs': {}
    }

    memory = np.zeros_like(target_labels, dtype=np.float64)
    identified = []
    for run_idx, seed in enumerate(range(RANDOM_STATE, RANDOM_STATE + N_RUNS)):
        logger.info(f"\n{'='*20} Run {seed}/{N_RUNS} {'='*20}")
        set_seed(seed)
        test_labels = []
        test_probs = []
        test_preds = []
        test_accs = []
        test_youden = []
        test_sens = []
        test_spec = []
        test_ids = []
        test_patient_ids = []
        test_thresholds = []

        for fold, (train_idx, test_idx) in enumerate(fold_splits):
            logger.info(f"\n{'='*20} Fold {fold+1}/{N_FOLDS} {'='*20}")
            if len(identified) > 0:
                NOISY_DROP = update_noisy_drop_variance(memory)
                # NOISY_DROP = update_noisy_drop_gmm(memory)
                drop_idx = np.random.permutation(identified)[:int(NOISY_DROP * len(identified))]
                train_idx = list(set(train_idx) - set(drop_idx))

            trainval_fold = df_train.iloc[train_idx].reset_index(drop=True)
            test_fold = df_train.iloc[test_idx].reset_index(drop=True)

            mask_proper, mask_cp = train.stratified_group_holdout(trainval_fold[hps.data.target_column].values, trainval_fold["participant"].values, 
                                                                    test_size=0.15, seed=seed)
            
            train_fold = trainval_fold.iloc[mask_proper].reset_index(drop=True)
            val_fold = trainval_fold.iloc[mask_cp].reset_index(drop=True)

            # Prepare data loaders
            train_loader, val_loader = train.prepare_fold_data(
                train_fold, val_fold, hps, fold, collate_fn,
                use_precomputed=args.use_precomputed,
                precomputed_dir=args.precomputed_dir
            )

            pool_model = pool_net(**hps.model)
            checkpoint_callback = ModelCheckpoint(
                dirpath=f"{hps.model_dir}/fold_{fold}",
                monitor="val/loss",
                filename=f"pool_fold{fold}_{{epoch:02d}}",
                save_top_k=1,
                mode="min",
            )

            tb_logger = TensorBoardLogger(save_dir=hps.model_dir, name=f"fold_{fold}", sub_dir="train")
            early_stopping = EarlyStopping(monitor="val/loss", patience=7, mode="min", verbose=False)
            runner_lightning = lightning_wrapper.CoughClassificationRunner(pool_model, hps=hps, custom_logger=logger, class_weights=[])
            trainer = L.Trainer(
                max_epochs=1000,
                callbacks=[checkpoint_callback, early_stopping],
                logger=tb_logger,
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                devices="auto",
                default_root_dir=hps.model_dir
            )

            trainer.fit(runner_lightning, train_dataloaders=train_loader, val_dataloaders=val_loader)
            
            runner_lightning.calibrate_threshold = True
            trainer.test(runner_lightning, dataloaders=val_loader)[0]
            optimized_threshold = runner_lightning.probs_threshold
            test_thresholds.append(optimized_threshold)

            # Prepare test data
            test_data = test_fold
            if args.use_precomputed and args.precomputed_dir:
                mapping_train = pd.read_csv(os.path.join(args.precomputed_dir, "feature_mapping_train.csv"))
                test_data = test_fold.merge(
                    mapping_train[['path_file', 'feature_path']], 
                    on='path_file', 
                    how='left'
                )
                feature_path_col = list(test_data.columns).index('feature_path')
            
            test_dataset = CoughDatasets(
                test_data.values, 
                hps.data,
                wav_stats_path=f"{hps.model_dir}/wav_stats.pickle" if not args.use_precomputed else None, 
                train=False,
                use_precomputed=args.use_precomputed
            )
            
            if args.use_precomputed and args.precomputed_dir:
                test_dataset.set_feature_path_column(feature_path_col)
            test_loader = DataLoader(
                test_dataset, 
                num_workers=28, 
                shuffle=False, 
                batch_size=hps.train.batch_size,
                pin_memory=True, 
                collate_fn=collate_fn
            )

            pool_model = pool_net(**hps.model)
            runner_lightning = lightning_wrapper.CoughClassificationRunner.load_from_checkpoint(
                trainer.checkpoint_callback.best_model_path,     
                model=pool_model,
                hps=hps, custom_logger=logger
            ).cuda()
            runner_lightning.eval()
            runner_lightning.probs_threshold = optimized_threshold

            fold_test_labels = []
            fold_test_probs = []
            fold_test_preds = []
            fold_test_patient_ids = []
            with torch.no_grad():
                for idx, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
                    _, audio, _, attention_masks, dse_ids, [patient_ids, _, _, _] = batch
                    audio = audio.cuda()
                    attention_masks = attention_masks.cuda()
                    out_model = runner_lightning.model.forward(audio, attention_mask=attention_masks)
                    logits = out_model['disease_logits']

                    probs = torch.sigmoid(logits).squeeze(-1)  # [B]
                    preds = (probs >= runner_lightning.probs_threshold).long().cpu().detach().numpy()
                    labels = torch.argmax(dse_ids, dim=1).cpu().detach().numpy()

                    fold_test_labels.extend(labels)
                    fold_test_probs.extend(probs.cpu().detach().numpy())
                    fold_test_preds.extend(preds)
                    fold_test_patient_ids.extend(patient_ids.cpu().detach().numpy())

            temp_labels = np.array(fold_test_labels)
            temp_preds = np.array(fold_test_preds)
            cm = confusion_matrix(temp_labels, temp_preds, labels=[0, 1])
            assert cm.shape == (2, 2), f"Expected binary confusion matrix, got {cm.shape}"
            TN, FP = cm[0, 0], cm[0, 1]
            FN, TP = cm[1, 0], cm[1, 1]
            sens = TP / (TP + FN) if (TP + FN) > 0 else 0.0   # TB recall
            spec = TN / (TN + FP) if (TN + FP) > 0 else 0.0   # non-TB recall
            b_acc = 0.5 * (sens + spec)
            youden = Youden_Weights[0] * sens + Youden_Weights[1] * spec - 1.0
            
            test_labels.append(fold_test_labels)
            test_probs.append(fold_test_probs)
            test_preds.append(fold_test_preds)
            test_accs.append(b_acc)
            test_youden.append(youden)
            test_sens.append(sens)
            test_spec.append(spec)
            test_ids.append(test_idx)
            test_patient_ids.append(fold_test_patient_ids)
            
        number_ids = [len(i) for i in test_ids]
        test_labels_all = np.concatenate(test_labels)
        pred_probs_all = np.concatenate(test_probs)
        test_ids_all = np.concatenate(test_ids)
        test_patient_ids_all = np.concatenate(test_patient_ids)
        threshold_all = np.concatenate([[test_thresholds[i]] * number_ids[i] for i in range(N_FOLDS)])

        test_accs_np = np.asarray(test_accs, dtype=np.float64)
        test_youden_np = np.asarray(test_youden, dtype=np.float64)

        weights_acc = np.max(test_accs_np) - test_accs_np # distance from best fold, distance = 0, larger value.
        weights_acc = safe_minmax_scale(weights_acc) # 0 - 1 Norm
        weights_acc = 1 - weights_acc # Best fold → weight ≈ 1
        weights_acc = np.concatenate([[weights_acc[i]]*number_ids[i] for i in range(N_FOLDS)])

        weights_youden = np.max(test_youden_np) - test_youden_np # distance from best fold, distance = 0, larger value.
        weights_youden = safe_minmax_scale(weights_youden) # 0 - 1 Norm
        weights_youden = 1 - weights_youden # Best fold → weight ≈ 1
        weights_youden = np.concatenate([[weights_youden[i]]*number_ids[i] for i in range(N_FOLDS)])
        
        #idx_temp = np.stack((np.arange(len(test_labels_all)), test_labels_all))
        #temp =  pred_probs_all[idx_temp[0,:], idx_temp[1,:]] # For Multiclass, Extract Only True Probs
        # “How confident is the model that the ground-truth label is correct?”
        temp = np.where(test_labels_all == 1, pred_probs_all, 1 - pred_probs_all)
        weights_probs_conf = safe_minmax_scale(temp)
        weights_probs_margin = safe_minmax_scale(np.abs(pred_probs_all - threshold_all))
        weights_probs = 0.7 * weights_probs_conf + 0.3 * weights_probs_margin

        # Adaptive objective weights: add more fold-level regularization after warmup.
        progress = min(1.0, run_idx / max(1, N_RUNS - 1))
        objective_weights = np.asarray(Objective_Function_Weights, dtype=np.float64).copy()
        objective_weights[1] = min(0.30, objective_weights[1] + 0.10 * progress)
        objective_weights[0] = max(0.45, objective_weights[0] - 0.10 * progress)
        objective_weights = objective_weights / objective_weights.sum()

        weights = (
            objective_weights[0] * weights_probs
            + objective_weights[1] * weights_acc
            + objective_weights[2] * weights_youden
        )
        order = np.argsort(test_ids_all)
        weights = weights[order]
        
        # Participant Aggregatiion
        test_patient_ids_all = test_patient_ids_all[order]
        _, inverse = np.unique(test_patient_ids_all, return_inverse=True)
        pid_sum = np.bincount(inverse, weights=weights)
        pid_cnt = np.bincount(inverse)
        pid_mean = pid_sum / pid_cnt
        weights = pid_mean[inverse]

        ema_keep = 0.9 if run_idx < WARMUP_RUNS else 0.7
        memory = (1.0 - ema_keep) * weights + ema_keep * memory

        TAU = update_tau_variance(TAU, memory) # 1) variance-driven
        #TAU = update_tau_validation(TAU, test_youden) # 2) validation-feedback (use Youden or b_acc)
        MEMORY_NOISE_THRES = estimate_noise_threshold_gmm(memory)

        fold_splits, fold_ids = sample_folds(N_FOLDS, memory, TAU)
        # fold_splits, fold_ids = sample_folds_grouped(
        #     N_FOLDS,
        #     memory,
        #     participant_ids=df_train["participant"].values,
        #     tau=TAU,
        #     random_state=seed
        # )
        identified, warmup_cap_ratio = select_identified_with_warmup(
            memory,
            MEMORY_NOISE_THRES,
            run_idx=run_idx,
            warmup_runs=WARMUP_RUNS,
            max_ratio=MAX_IDENTIFIED_RATIO,
            cap_after_warmup=CAP_AFTER_WARMUP,
        )
        np.save(f"{hps.model_dir}/memory.npy", memory)
        monitor_metrics['runs'][seed] = {
            'test_accs': test_accs,
            'test_sens': test_sens,
            'test_spec': test_spec,
            'test_youden': test_youden,
            'confidence': memory,
            'TAU': TAU,
            'MEMORY_NOISE_THRES': MEMORY_NOISE_THRES,
            'NOISY_DROP': NOISY_DROP,
            'objective_weights': objective_weights,
            'warmup_cap_ratio': warmup_cap_ratio,
            'identified_ratio': len(identified) / len(df_train),
            'percentage_datasets': 1 - round(len(identified) / len(df_train), 2),
            'percentage_positivedata': round((train_fold[hps.data.target_column] == 1).sum() / len(df_train), 2),
            'percentage_negativedata': round((train_fold[hps.data.target_column] == 0).sum() / len(df_train), 2),
        }
        with open(os.path.join(hps.model_dir, "info_recov.pkl"), "wb") as f:
            pickle.dump(monitor_metrics, f)

        metrics = ['test_accs', 'test_sens', 'test_spec', 'test_youden', 'confidence', 'TAU', 'MEMORY_NOISE_THRES', 'NOISY_DROP',
                   'identified_ratio', 'warmup_cap_ratio', 'percentage_datasets', 'percentage_positivedata', 'percentage_negativedata']
        for metric in metrics:
            values_mean = []
            for runs, values in monitor_metrics['runs'].items():
                values_mean.append(np.mean(values[metric]))
            plt.figure()
            plt.plot(values_mean)
            plt.xlabel("Run Index")
            plt.ylabel(metric)
            plt.title(f"{metric} across runs")
            plt.grid(True)
            plt.savefig(f"{hps.model_dir}/{metric}_across_runs.png", dpi=300, bbox_inches="tight")
            plt.close()
        
    identified, _ = select_identified_with_warmup(
        memory,
        MEMORY_NOISE_THRES,
        run_idx=N_RUNS - 1,
        warmup_runs=WARMUP_RUNS,
        max_ratio=MAX_IDENTIFIED_RATIO,
        cap_after_warmup=CAP_AFTER_WARMUP,
    )
    np.save(f"{hps.model_dir}/identified.npy", identified)
    with open(os.path.join(hps.model_dir, "info_recov.pkl"), "wb") as f:
        pickle.dump(monitor_metrics, f)
    

if __name__ == "__main__":
    main()
