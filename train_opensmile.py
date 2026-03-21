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
current_module = sys.modules[__name__]
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

def sample_folds(n_folds:int, weights:np.array, tau:float, **kwargs) -> Tuple[List,Dict]:
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

def sample_folds_grouped(n_folds: int, weights: np.ndarray, tau: float, participant_ids: np.ndarray) -> Tuple[List, Dict]:
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
def estimate_noise_threshold_gmm(memory, alpha=0.5):
    """
    Fit 2-component Gaussian mixture and
    return intersection midpoint between means.
    < 0.5 shifts toward clean_mean
    """
    mem = memory.reshape(-1, 1)

    gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=0)
    gmm.fit(mem)

    means = gmm.means_.flatten()
    order = np.argsort(means)

    noisy_mean = means[order[0]]
    clean_mean = means[order[1]]

    # midpoint between clusters (robust + simple)
    #threshold = float((noisy_mean + clean_mean) / 2.0)
    threshold = alpha * noisy_mean + (1 - alpha) * clean_mean
    return threshold

def update_memory_alpha(weights: np.ndarray,
    memory: np.ndarray, prev_alpha: float = 0.3,
    alpha_min: float = 0.10, alpha_max: float = 0.60,
    k: float = 5.0, smooth: float = 0.8) -> float:
    """
    Adaptive EMA alpha:
    - larger drift |weights - memory| -> larger alpha (faster update)
    - smaller drift -> smaller alpha (more smoothing)
    """
    drift = float(np.mean(np.abs(weights - memory)))  # in [0,1] after your min-max
    raw_alpha = alpha_min + (alpha_max - alpha_min) * (1.0 - np.exp(-k * drift))
    alpha = smooth * prev_alpha + (1.0 - smooth) * raw_alpha
    return float(np.clip(alpha, alpha_min, alpha_max))

def memory_alpha_fixated(weights: np.ndarray, memory: np.ndarray, prev_alpha: float = 0.3, **kwargs) -> float:
    return 0.3

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

    labels_temp = torch.tensor(target_labels.values)
    num_pos = (labels_temp == 1).sum()
    num_neg = (labels_temp == 0).sum()
    pos_weight = num_neg / (num_pos + 1e-8)

    if not args.use_precomputed:
        if hps.data.acoustic_feature and hps.data.mean_std_norm:
            utils.compute_spectrogram_stats_from_dataset(
                df_train, hps.data, 
                pickle_path=f"{hps.model_dir}/wav_stats.pickle"
            )
        else:
            utils.compute_wav_stats(
                df_train, "path_file", 
                pickle_path=f"{hps.model_dir}/wav_stats.pickle"
            )
    
    df_cirdz = pd.read_csv(f'/run/media/fourier/Data1/Pras/Thesis_Nexus/NXSCough/data/metadata_cirdz.csv.train')
    df_cirdz = df_cirdz.reset_index(drop=True)
    df_cirdz = df_cirdz[hps.data.column_order]
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
    if args.use_tensorboard:
        logger.info(f"✨ Tensorboard: http://100.101.198.75:{port}/#scalars&_smoothingWeight=0")
    logger.info(f"======================================")

    pool_net = setup_model(hps, is_init=args.init)
    # =============================================================
    # SECTION: Setup Logger, Dataloader
    # =============================================================
    fold_thr = []
    oof_p = np.zeros(len(df_train), dtype=float)
    oof_label = np.zeros(len(df_train), dtype=float)
    
    splitter, num_folds = create_data_split(df_train, target_labels, use_kfold=hps.train.use_Kfold, n_splits=10, random_state=RNG_SEED)
    for fold_outter, (inner_idx, test_idx) in enumerate(splitter):
        logger.info(f"\n{'='*20} Fold {fold_outter+1}/{num_folds} {'='*20}")
        
        # proportion of positive labels (class 1) Train: 0.30 30% of training samples are label=1 (positive/TB)
        train_labels = target_labels.iloc[inner_idx]
        test_labels = target_labels.iloc[test_idx]
        print(f"Fold {fold_outter+1} | Train: {train_labels.mean():.2f} | Val: {test_labels.mean():.2f}")

        inner_fold = df_train.iloc[inner_idx].reset_index(drop=True)
        test_fold = df_train.iloc[test_idx].reset_index(drop=True)

        mask_proper, mask_cp = stratified_group_holdout(inner_fold[hps.data.target_column].values, inner_fold["participant"].values, 
                                                                  test_size=0.15, seed=RNG_SEED + fold_outter)

        train_fold = inner_fold.iloc[mask_proper].reset_index(drop=True)
        val_fold = inner_fold.iloc[mask_cp].reset_index(drop=True)
            
        train_loader, val_loader = prepare_fold_data(
            train_fold, val_fold, hps, collate_fn,
            use_precomputed=args.use_precomputed,
            precomputed_dir=args.precomputed_dir
        )

        pool_model = pool_net(**hps.model)
        checkpoint_callback = ModelCheckpoint(
            dirpath=f"{hps.model_dir}/{fold_outter}",
            monitor="val/loss",
            filename=f"pool_fold{fold_outter}_{{epoch:02d}}",
            save_top_k=1,
            mode="min",
        )

        tb_logger = TensorBoardLogger(save_dir=hps.model_dir, name=f"{fold_outter}", sub_dir="train")
        early_stopping = EarlyStopping(monitor="val/loss", patience=7, mode="min", verbose=False)
        runner_lightning = lightning_wrapper.CoughClassificationRunner(pool_model, hps=hps, custom_logger=logger, class_weights=pos_weight)
        trainer = L.Trainer(
            max_epochs=1000,
            callbacks=[checkpoint_callback, early_stopping],
            logger=tb_logger,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices="auto",
            default_root_dir=hps.model_dir,
            deterministic=True,
        )

        trainer.fit(runner_lightning, train_dataloaders=train_loader, val_dataloaders=val_loader)
        # runner_lightning.test_raw = True
        # trainer.test(runner_lightning, dataloaders=val_loader, ckpt_path="best")
        # oof_probs = runner_lightning.test_outputs['probs'].reshape(-1)
        # iso = IsotonicRegression(out_of_bounds="clip")
        # iso.fit(oof_probs, val_fold[hps.data.target_column].values)
        # payload = {"calibrator": iso}
        # with open(os.path.join(f"{hps.model_dir}/{fold_outter}", "calibrator.pkl"), "wb") as f:
        #     pickle.dump(payload, f)

        production_path = os.path.join(f"{hps.model_dir}/{fold_outter}", "best_model.ckpt")
        shutil.move(trainer.checkpoint_callback.best_model_path, production_path)

        trainer = L.Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu", devices="auto")
        runner_lightning = lightning_wrapper.CoughClassificationRunner.load_from_checkpoint(
            os.path.join(f"{hps.model_dir}/{fold_outter}", "best_model.ckpt"),
            model=pool_model, hps=hps, custom_logger=logger, class_weights=pos_weight)
        runner_lightning.eval()

        runner_lightning.calibrate_threshold = True
        trainer.test(runner_lightning, dataloaders=val_loader)

        results_dict = evaluate_on_dataset(runner_lightning, trainer, test_fold, hps, collate_fn,
                                            use_precomputed=args.use_precomputed,
                                            precomputed_dir=args.precomputed_dir)
        write_results_to_file("result_overall.txt", results_dict, f"{hps.model_dir}", {0: "Test " + str(fold_outter)})
        
        oof_p[test_idx] = results_dict[0]["raw_data"]['probs'].reshape(-1)
        oof_label[test_idx] = results_dict[0]["raw_data"]['labels'].reshape(-1)
        fold_thr.append(runner_lightning.probs_threshold)
        
        results_dict = {}
        loader = DataLoader(
            CoughDatasets(
                df_cirdz.values, 
                hps.data,
                wav_stats_path=f"{args.precomputed_dir}/wav_stats.pickle", 
                train=False,
                use_precomputed=False
            ), 
            num_workers=28, 
            shuffle=False, 
            batch_size=hps.train.batch_size,
            pin_memory=True, 
            collate_fn=collate_fn
        )
        results = trainer.test(runner_lightning, dataloaders=loader)[0]
        results_dict[0] = {
            "metrics": results,
            "raw_data": runner_lightning.test_outputs,
        }
        write_results_to_file("result_overall.txt", results_dict, f"{hps.model_dir}", {0: "CIRDZ " + str(fold_outter)})

        if os.path.exists(production_path):
            os.remove(production_path)

    with open(f"{hps.model_dir}/info_run.pkl", "wb") as f:
        pickle.dump({
            "fold_thr": fold_thr, 
            "oof_p": oof_p, 
            "oof_label": oof_label, 
            }, f)

    if tb_process is not None:
        tb_process.terminate()
        try:
            tb_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            tb_process.kill()
    
    # if not args.eval and hps.train.use_Kfold:
    #     best_fold_idx, production_path = select_best_fold(fold_metrics, fold_checkpoints, hps.model_dir, logger)
    #     save_fold_info(best_fold_idx, fold_metrics, fold_thresholds, hps.model_dir)
    # elif not args.eval:
    #     best_fold_idx = 0
    #     best_model_path = fold_checkpoints[best_fold_idx]
    #     production_path = os.path.join(hps.model_dir, "best_model.ckpt")
    #     shutil.copy2(best_model_path, production_path)
    #     save_fold_info(best_fold_idx, fold_metrics, fold_thresholds, hps.model_dir)
    # else:
    #     # In eval mode, load best_fold_idx from existing info_fold.pkl
    #     info_fold_data = load_fold_info(hps.model_dir)
    #     best_fold_idx = info_fold_data.get("best_fold_idx", 0)

    # # =============================================================
    # # SECTION: Test Phase
    # # =============================================================
    # db_map = {
    #     0: "TBCoda Logitudinal",
    #     1: "TBCoda Solicited",
    #     2: "TBScreen Logitudinal",
    #     3: "TBScreen Solicited",
    #     4: "CIRDZ",
    #     5: "UK19Covid",
    # }

    # runner_lightning = lightning_wrapper.CoughClassificationRunner.load_from_checkpoint(
    #     os.path.join(hps.model_dir, "best_model.ckpt"),
    #     model=pool_model,
    #     hps=hps, custom_logger=logger
    # )
    # runner_lightning.eval()
    # # runner_lightning.calibrator = iso
    # trainer = L.Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu", devices="auto")

    # info_fold = load_fold_info(hps.model_dir)
    # best_fold_idx = info_fold["best_fold_idx"]
    # fold_metrics = info_fold["fold_metrics"]
    # runner_lightning.probs_threshold = info_fold['best_threshold']

    # ###### Calibrate Threshold or Load if Exist
    # # if os.path.exists(os.path.join(model_dir, "probs_threshold.pkl")):
    # #     with open(os.path.join(model_dir, "probs_threshold.pkl"), "rb") as f:
    # #         runner_lightning.probs_threshold = pickle.load(f)['probs_threshold']
    # # else:
    # #     splitter, num_folds = create_data_split(df_train, target_labels, use_kfold=hps.train.use_Kfold)
    # #     train_idx, val_idx = list(splitter)[best_fold_idx]

    # #     train_fold = df_train.iloc[train_idx].reset_index(drop=True)
    # #     val_fold = df_train.iloc[val_idx].reset_index(drop=True)
    # #     train_loader, val_loader = prepare_fold_data(train_fold, val_fold, hps, best_fold_idx, collate_fn)

    # #     runner_lightning.calibrate_threshold = True
    # #     results = trainer.test(runner_lightning, dataloaders=val_loader)[0]
    # #     runner_lightning.calibrate_threshold = False
        
    # #     payload = {
    # #         "probs_threshold": runner_lightning.probs_threshold,
    # #     }
    # #     with open(os.path.join(model_dir, "probs_threshold.pkl"), "wb") as f:
    # #         pickle.dump(payload, f)

    # ############################################# Overall Report #########################################################
    # with open(f"{model_dir}/result_overall.txt", "w") as f:
    #     f.write(f"")

    # df_train_eval = pd.read_csv(f'/run/media/fourier/Data1/Pras/Thesis_Nexus/NXSCough/data/{hps.data.metadata_csv}.train')
    # df_train_eval = df_train_eval.reset_index(drop=True)
    # df_train_eval = df_train_eval[hps.data.column_order]

    # results_dict = evaluate_on_dataset(runner_lightning, trainer, df_train_eval, hps, best_fold_idx, collate_fn)
    # write_results_to_file("result_overall.txt", results_dict, model_dir, {0: "Train"})

    # df_test_eval = pd.read_csv(f'/run/media/fourier/Data1/Pras/Thesis_Nexus/NXSCough/data/{hps.data.metadata_csv}.test')
    # df_test_eval = df_test_eval.reset_index(drop=True)
    # df_test_eval = df_test_eval[hps.data.column_order]

    # runner_lightning.generate_figure = True
    # results_dict = evaluate_on_dataset(runner_lightning, trainer, df_test_eval, hps, best_fold_idx, collate_fn)
    # write_results_to_file("result_overall.txt", results_dict, model_dir, {0: "Test"})
    # runner_lightning.generate_figure = False

    # df_cirdz = pd.read_csv(f'/run/media/fourier/Data1/Pras/Thesis_Nexus/NXSCough/data/cirdz.csv.test')
    # df_cirdz = df_cirdz.reset_index(drop=True)
    # df_cirdz = df_cirdz[hps.data.column_order]

    # results_dict = evaluate_on_dataset(runner_lightning, trainer, df_cirdz, hps, best_fold_idx, collate_fn)
    # write_results_to_file("result_overall.txt", results_dict, model_dir, {0: "Unseen"})
    # ############################################# PerDB Report ###########################################################
    # # Handle result_summary.txt versioning
    # rotate_result_summary(model_dir)
    # with open(f"{model_dir}/result_summary.txt", "w") as f:
    #     f.write(f"{'='*25} Train Phase {'='*25}\n")
    # df_train_eval = pd.read_csv(f'/run/media/fourier/Data1/Pras/Thesis_Nexus/NXSCough/data/{hps.data.metadata_csv}.train')
    # df_train_eval = df_train_eval.reset_index(drop=True)
    # df_train_eval = df_train_eval[hps.data.column_order + ['db']]

    # results_dict = evaluate_on_dataset(runner_lightning, trainer, df_train_eval, hps, best_fold_idx, collate_fn, db_column='db')
    # write_results_to_file("result_summary.txt", results_dict, model_dir, db_map)

    # with open(f"{model_dir}/result_summary.txt", "a") as f:
    #     f.write(f"\n{'='*25} Test Phase {'='*25}\n")
    # df_test_eval = pd.read_csv(f'/run/media/fourier/Data1/Pras/Thesis_Nexus/NXSCough/data/{hps.data.metadata_csv}.test')
    # df_test_eval = df_test_eval.reset_index(drop=True)
    # df_test_eval = df_test_eval[hps.data.column_order + ['db']]

    # results_dict = evaluate_on_dataset(runner_lightning, trainer, df_test_eval, hps, best_fold_idx, collate_fn, db_column='db')
    # write_results_to_file("result_summary.txt", results_dict, model_dir, db_map)

    # df_cirdz = pd.read_csv(f'/run/media/fourier/Data1/Pras/Thesis_Nexus/NXSCough/data/cirdz.csv.test')
    # df_cirdz = df_cirdz.reset_index(drop=True)
    # df_cirdz = df_cirdz[hps.data.column_order]

    # results_dict = evaluate_on_dataset(runner_lightning, trainer, df_cirdz, hps, best_fold_idx, collate_fn)
    # write_results_to_file("result_summary.txt", results_dict, model_dir, {0: "CIRDZ"})

    # # =============================================================
    # # SECTION: Cleaning
    # # =============================================================
    # cleanup_fold_directories(hps.model_dir, best_fold_idx)

if __name__ == "__main__":
    main()
