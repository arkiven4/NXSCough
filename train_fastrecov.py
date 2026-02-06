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
import re

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
from sklearn.model_selection import KFold

# Local imports
import commons
import lightning_wrapper
import models
import utils
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

class TeeFiltered:
    def __init__(self, terminal, logfile):
        self.terminal = terminal
        self.logfile = logfile
        self.buffer = ""

        # pattern for tqdm / lightning progress lines
        self.progress_pattern = re.compile(r"^Epoch \d+:\s+\d+%")

    def write(self, data):
        self.terminal.write(data)
        self.terminal.flush()

        # split into real lines
        self.buffer += data
        lines = self.buffer.split("\n")
        self.buffer = lines.pop()

        for line in lines:
            # skip progress-bar redraw lines
            if "\r" in line or self.progress_pattern.match(line):
                continue

            self.logfile.write(line + "\n")
            self.logfile.flush()

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()

def load_config(config_path, model_dir, args):
    """Load and override configuration from JSON file."""
    with open(config_path) as f:
        config = json.load(f)
    
    # Centralized override map
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
    
    # Save updated config
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    hps = utils.HParams(**config)
    hps.model_dir = model_dir
    hps.data.mae_training = hps.train.mae_training
    hps.data.ssccl_training = hps.train.ssccl_training
    
    return hps


def load_data(hps):
    """Load training and test dataframes."""
    df_train = pd.read_csv(f'/run/media/fourier/Data1/Pras/Thesis_Nexus/NXSCough/data/{hps.data.metadata_csv}.train')
    df_train = df_train.reset_index(drop=True)
    df_train = df_train[hps.data.column_order]

    try:
        df_test = pd.read_csv(f'/run/media/fourier/Data1/Pras/Thesis_Nexus/NXSCough/data/{hps.data.metadata_csv}.test')
        df_test = df_test.reset_index(drop=True)
        df_test = df_test[hps.data.column_order]
    except:
        df_test = None
    
    return df_train, df_test


def get_collate_fn(hps):
    """Get appropriate collate function based on model type."""
    if "qwen" in hps.model.pooling_model.lower():
        from transformers import Qwen3OmniMoeProcessor
        collate_fn = CoughDatasetsProcessorCollate(
            hps.data.many_class,
            processor=Qwen3OmniMoeProcessor.from_pretrained(
                "/run/media/fourier/Data1/Pras/pretrain_models/Qwen3-Omni-30B-A3B-Thinking"
            ),
            sampling_rate=hps.data.sampling_rate
        )
    else:
        collate_fn = CoughDatasetsCollate(hps.data.many_class)
    
    return collate_fn


def setup_model(hps, is_init=True):
    """Load or initialize model based on configuration."""
    hps.model.spk_dim = 0
    if is_init:
        pool_net = getattr(models, hps.model.pooling_model)
        pool_model = pool_net(hps.model.feature_dim, **hps.model)
        shutil.copy2('./models.py', f'{hps.model_dir}/model_net.py.bak')
    else:
        temp_path = tempfile.NamedTemporaryFile(suffix=".py", delete=False).name
        shutil.copy(f"{hps.model_dir}/model_net.py.bak", temp_path)
        spec = importlib.util.spec_from_file_location("model_net", temp_path)
        model_modules = importlib.util.module_from_spec(spec)
        sys.modules["model_net"] = model_modules
        spec.loader.exec_module(model_modules)
        pool_net = getattr(model_modules, hps.model.pooling_model)
        pool_model = pool_net(hps.model.feature_dim, **hps.model)
    
    return pool_net, pool_model


def create_data_split(df_train, target_labels, use_kfold=True, n_splits=5, test_size=0.2, random_state=42):
    """Create train/validation splits using K-Fold or simple split."""
    if use_kfold:
        sgkf = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state
        )
        splitter = sgkf.split(
            X=df_train,
            y=target_labels,
            groups=df_train["participant"]
        )
        num_folds = sgkf.get_n_splits()

        # skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        # splitter = skf.split(df_train, target_labels)
        # num_folds = skf.get_n_splits()
    else:
        train_idx, val_idx = train_test_split(
            df_train.index.to_numpy(),
            test_size=test_size,
            random_state=random_state,
            stratify=target_labels
        )
        splitter = [(train_idx, val_idx)]
        num_folds = 1
    
    return splitter, num_folds


def create_sampler(train_fold, hps):
    """Create appropriate sampler for training data."""
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
        
        pos_weight = 1.0 / num_pos
        neg_weight = 1.0 / num_neg
        
        sample_weights = torch.zeros(len(train_fold), dtype=torch.double)
        sample_weights[positive_idx] = pos_weight
        sample_weights[negative_idx] = neg_weight
        
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        # patient_ids = train_fold['participant'].astype(str).values
        # labels = train_fold[hps.data.target_column].astype(int).values
        # sampler = AutoPatientBatchSampler(
        #     labels=labels,
        #     patient_ids=patient_ids,
        #     target_batch_wavs=hps.train.batch_size,
        # )
    
    return sampler


def prepare_fold_data(train_fold, val_fold, hps, fold, collate_fn):
    """Prepare datasets and dataloaders for a fold."""
    # Compute statistics
    if hps.data.acoustic_feature and hps.data.mean_std_norm:
        utils.compute_spectrogram_stats_from_dataset(
            train_fold, 
            hps.data, 
            pickle_path=f"{hps.model_dir}/wav_stats.pickle"
        )
    else:
        utils.compute_wav_stats(
            train_fold, 
            "path_file", 
            pickle_path=f"{hps.model_dir}/wav_stats.pickle"
        )
    
    # Create datasets
    train_dataset = CoughDatasets(
        train_fold.values, 
        hps.data,
        wav_stats_path=f"{hps.model_dir}/wav_stats.pickle", 
        train=True
    )
    val_dataset = CoughDatasets(
        val_fold.values, 
        hps.data,
        wav_stats_path=f"{hps.model_dir}/wav_stats.pickle", 
        train=False
    )
    
    # Create sampler
    sampler = create_sampler(train_fold, hps)
    #sampler = None
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        num_workers=28, 
        sampler=sampler, 
        batch_size=hps.train.batch_size,
        #batch_sampler=sampler,
        pin_memory=True, 
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        num_workers=28, 
        shuffle=False, 
        batch_size=hps.train.batch_size,
        pin_memory=True, 
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader


def select_best_fold(fold_metrics, fold_checkpoints, model_dir, logger):
    """Select the best fold based on balanced accuracy closest to mean."""
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
    
    production_path = None
    if best_model_path and os.path.exists(best_model_path):
        production_path = os.path.join(model_dir, "best_model.ckpt")
        shutil.copy2(best_model_path, production_path)
        logger.info(f"🏆 Saved Production Model to: {production_path}")
    else:
        logger.info("❌ Could not find best model checkpoint to copy.")
    
    return best_fold_idx, production_path


def save_fold_info(best_fold_idx, fold_metrics, fold_thresholds, model_dir):
    """Save fold information to pickle file."""
    payload = {
        "best_fold_idx": best_fold_idx,
        "best_threshold": fold_thresholds[best_fold_idx],
        "fold_metrics": fold_metrics,
    }
    
    with open(os.path.join(model_dir, "info_fold.pkl"), "wb") as f:
        pickle.dump(payload, f)


def load_fold_info(model_dir):
    """Load fold information from pickle file."""
    info_path = os.path.join(model_dir, "info_fold.pkl")
    if os.path.exists(info_path):
        with open(info_path, "rb") as f:
            return pickle.load(f)
    return {"best_fold_idx": 0, "best_threshold": 0.0, "fold_metrics": []}


def rotate_result_summary(model_dir):
    """Rotate existing result summary files (keep last 3 versions)."""
    result_summary_path = f"{model_dir}/result_summary.txt"
    if os.path.exists(result_summary_path):
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


def evaluate_on_dataset(runner_lightning, trainer, df, hps, best_fold_idx, collate_fn, db_column=None):
    """Evaluate model on dataset, optionally grouped by database type."""
    results_dict = {}
    
    if db_column and db_column in df.columns:
        for db_type in df[db_column].unique().tolist():
            df_subset = df[df[db_column] == db_type]
            dataset = CoughDatasets(
                df_subset.values, 
                hps.data,
                wav_stats_path=f"{hps.model_dir}/wav_stats_fold_{best_fold_idx}.pickle", 
                train=False
            )
            loader = DataLoader(
                dataset, 
                num_workers=28, 
                shuffle=False,
                batch_size=hps.train.batch_size, 
                pin_memory=True, 
                collate_fn=collate_fn
            )
            results = trainer.test(runner_lightning, dataloaders=loader)[0]
            results_dict[db_type] = results
    else:
        dataset = CoughDatasets(
            df.values, 
            hps.data,
            wav_stats_path=f"{hps.model_dir}/wav_stats_fold_{best_fold_idx}.pickle", 
            train=False
        )
        loader = DataLoader(
            dataset, 
            num_workers=28, 
            shuffle=False, 
            batch_size=hps.train.batch_size,
            pin_memory=True, 
            collate_fn=collate_fn
        )
        results = trainer.test(runner_lightning, dataloaders=loader)[0]
        results_dict[0] = results
    
    return results_dict


def write_results_to_file(filename, results_dict, model_dir, db_map=None):
    """Write evaluation results to summary file."""
    with open(f"{model_dir}/{filename}", "a") as f:
        for db_type, results in results_dict.items():
            db_name = db_map.get(db_type, f"Database {db_type}") if db_map else "Full Dataset"
            f.write(
                f"{db_name} - "
                f"Acc {results['test_acc']:.4f} | "
                f"BalAcc {results['test_bacc']:.4f} | "
                f"Sens {results['test_sens']:.4f} | "
                f"Spec {results['test_spec']:.4f} | "
                f"AUROC {results['test_auroc']:.4f} | "
                f"pAUROC {results['test_pauroc']:.4f}\n"
            )


def cleanup_fold_directories(model_dir, best_fold_idx=None):
    """Remove fold directories after best model is saved."""
    if os.path.isfile(os.path.join(model_dir, "best_model.ckpt")):
        keep_name = None
        if best_fold_idx is not None:
            keep_name = f"fold_{best_fold_idx}"

        # 9.ckpt
        for name in os.listdir(model_dir):
            p = os.path.join(model_dir, name)
            if os.path.isdir(p) and name.startswith("fold_"):
                if keep_name is not None and name == keep_name:
                    ckpts = glob.glob(os.path.join(p, "*.ckpt"))
                    for ckpt in ckpts:
                        os.remove(ckpt)
                    print(f"Kept: {p}")
                    continue
                shutil.rmtree(p)
                print(f"Removed: {p}")
    else:
        print("best_model.ckpt not found; no folders removed.")

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

from typing import Dict, List, Tuple
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
    hps = load_config(config_path, model_dir, args)

    logterminal_file = open(os.path.join(model_dir, "log_terminal.txt"), "w")
    sys.stdout = TeeFiltered(sys.__stdout__, logterminal_file)
    sys.stderr = TeeFiltered(sys.__stderr__, logterminal_file)
    # =============================================================
    # SECTION: Loading Data
    # =============================================================
    df_train, _ = load_data(hps)
    collate_fn = get_collate_fn(hps)
    target_labels = df_train[hps.data.target_column]

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

    pool_net, pool_model = setup_model(hps, is_init=args.init)
    # =============================================================
    # SECTION: Setup Logger, Dataloader
    # =============================================================
    N_RUNS = 50
    N_FOLDS = 5
    TAU = 0.1
    RANDOM_STATE = 1
    MEMORY_NOISE_THRES = 0.4
    NOISY_DROP = 0.75 # Dropping bottom N * NOISY_DROP of the dataset 
    Objective_Function_Weights = [1.0, 0.0, 0.0]

    kf = KFold(n_splits=N_FOLDS, random_state=RANDOM_STATE, shuffle=True)
    fold_splits = list(kf.split(df_train, target_labels))

    monitor_metrics = {
        'N_RUNS': N_RUNS,
        'N_FOLDS': N_FOLDS,
        'TAU': TAU,
        'MEMORY_NOISE_THRES': MEMORY_NOISE_THRES,
        'NOISY_DROP': NOISY_DROP,
        'Objective_Function_Weights': Objective_Function_Weights,
        'runs': {}
    }
    memory = np.zeros_like(target_labels)
    identified = []
    for seed in range(RANDOM_STATE, RANDOM_STATE + N_RUNS):
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

        for fold, (train_idx, test_idx) in enumerate(fold_splits):
            logger.info(f"\n{'='*20} Fold {fold+1}/{N_FOLDS} {'='*20}")
            if len(identified) > 0:
                drop_idx = np.random.permutation(identified)[:int(NOISY_DROP * len(identified))]
                train_idx = list(set(train_idx) - set(drop_idx))

            trainval_fold = df_train.iloc[train_idx].reset_index(drop=True)
            test_fold = df_train.iloc[test_idx].reset_index(drop=True)

            sgkf_fold = StratifiedGroupKFold(n_splits=5, shuffle=True)
            trainfold_idx, valfold_idx = next(sgkf_fold.split(trainval_fold, trainval_fold[hps.data.target_column], trainval_fold["participant"]))
            train_fold = trainval_fold.iloc[trainfold_idx].reset_index(drop=True)
            val_fold  = trainval_fold.iloc[valfold_idx].reset_index(drop=True)

            # Prepare data loaders
            train_loader, val_loader = prepare_fold_data(train_fold, val_fold, hps, fold, collate_fn)

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
            runner_lightning.calibrate_threshold = False
            optimized_threshold = runner_lightning.probs_threshold

            test_dataset = CoughDatasets(
                test_fold.values, 
                hps.data,
                wav_stats_path=f"{hps.model_dir}/wav_stats.pickle", 
                train=False
            )
            test_loader = DataLoader(
                test_dataset, 
                num_workers=28, 
                shuffle=False, 
                batch_size=hps.train.batch_size,
                pin_memory=True, 
                collate_fn=collate_fn
            )

            pool_model = pool_net(hps.model.feature_dim, **hps.model)
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
                    wavnames, audio, _, attention_masks, dse_ids, [patient_ids, _, _, _] = batch
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
            youden = sens + spec - 1.0
            
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

        weights_acc = np.max(test_accs) - test_accs # distance from best fold, distance = 0, larger value.
        weights_acc = (weights_acc - weights_acc.min()) / (weights_acc.max()-weights_acc.min()) # 0 - 1 Norm
        weights_acc = 1 - weights_acc # Best fold → weight ≈ 1
        weights_acc = np.concatenate([[weights_acc[i]]*number_ids[i] for i in range(N_FOLDS)])

        weights_youden = np.max(test_youden) - test_youden # distance from best fold, distance = 0, larger value.
        weights_youden = (weights_youden - weights_youden.min()) / (weights_youden.max()-weights_youden.min()) # 0 - 1 Norm
        weights_youden = 1 - weights_youden # Best fold → weight ≈ 1
        weights_youden = np.concatenate([[weights_youden[i]]*number_ids[i] for i in range(N_FOLDS)])
        
        #idx_temp = np.stack((np.arange(len(test_labels_all)), test_labels_all))
        #temp =  pred_probs_all[idx_temp[0,:], idx_temp[1,:]] # For Multiclass, Extract Only True Probs
        # “How confident is the model that the ground-truth label is correct?”
        temp = np.where(test_labels_all == 1, pred_probs_all, 1 - pred_probs_all)
        #weights = 1 - (temp.max() - temp)
        weights_probs = (temp - temp.min()) / (temp.max() - temp.min() + 1e-8)
        weights = Objective_Function_Weights[0] * weights_probs + Objective_Function_Weights[1] * weights_acc + Objective_Function_Weights[2] * weights_youden
        order = np.argsort(test_ids_all)
        weights = weights[order]
        
        # Participant Aggregatiion
        test_patient_ids_all = test_patient_ids_all[order]
        _, inverse = np.unique(test_patient_ids_all, return_inverse=True)
        pid_sum = np.bincount(inverse, weights=weights)
        pid_cnt = np.bincount(inverse)
        pid_mean = pid_sum / pid_cnt
        weights = pid_mean[inverse]

        memory = 0.3 * weights + 0.7 * memory

        fold_splits, fold_ids = sample_folds(N_FOLDS, memory, TAU)
        identified = np.where(memory<=MEMORY_NOISE_THRES)[0]
        np.save(f"{hps.model_dir}/memory.npy", memory)
        monitor_metrics['runs'][seed] = {
            'test_accs': test_accs,
            'test_sens': test_sens,
            'test_spec': test_spec,
            'test_youden': test_youden,
            'confidence': memory,

        }
        with open(os.path.join(hps.model_dir, "info_recov.pkl"), "wb") as f:
            pickle.dump(monitor_metrics, f)

        metrics = ['test_accs', 'test_sens', 'test_spec', 'confidence']
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
        
    identified = np.where(memory<=MEMORY_NOISE_THRES)[0]
    np.save(f"{hps.model_dir}/identified.npy", identified)
    with open(os.path.join(hps.model_dir, "info_recov.pkl"), "wb") as f:
        pickle.dump(monitor_metrics, f)
    

if __name__ == "__main__":
    main()
