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
import losses
import models
import utils
from cough_datasets import (
    CoughDatasets,
    CoughDatasetsCollate,
    CoughDatasetsProcessorCollate,
    CoughDetectionRatioBatchSampler,
    CoughDiseaseBinaryBatchSampler,
    PatientBatchSampler
)

torch.set_float32_matmul_precision("medium")
cmap = cm.get_cmap("viridis")

#######################################################################
# REUSABLE FUNCTIONS
#######################################################################

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
    df_test = pd.read_csv(f'/run/media/fourier/Data1/Pras/Thesis_Nexus/NXSCough/data/{hps.data.metadata_csv}.test')
    
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    
    df_train = df_train[hps.data.column_order]
    df_test = df_test[hps.data.column_order]
    
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
        # sampler = PatientBatchSampler(
        #     patient_ids=patient_ids,
        #     patients_per_batch=hps.train.batch_size // 2,
        #     coughs_per_patient=2,
        # )
    
    return sampler


def prepare_fold_data(train_fold, val_fold, hps, fold, collate_fn):
    """Prepare datasets and dataloaders for a fold."""
    # Compute statistics
    if hps.data.acoustic_feature and hps.data.mean_std_norm:
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
    
    # Create datasets
    train_dataset = CoughDatasets(
        train_fold.values, 
        hps.data,
        wav_stats_path=f"{hps.model_dir}/wav_stats_fold_{fold}.pickle", 
        train=True
    )
    val_dataset = CoughDatasets(
        val_fold.values, 
        hps.data,
        wav_stats_path=f"{hps.model_dir}/wav_stats_fold_{fold}.pickle", 
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


def save_fold_info(best_fold_idx, fold_metrics, model_dir):
    """Save fold information to pickle file."""
    payload = {
        "best_fold_idx": best_fold_idx,
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
    return {"best_fold_idx": 0, "fold_metrics": []}


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

        for name in os.listdir(model_dir):
            p = os.path.join(model_dir, name)
            if os.path.isdir(p) and name.startswith("fold_"):
                if keep_name is not None and name == keep_name:
                    print(f"Kept: {p}")
                    continue
                shutil.rmtree(p)
                print(f"Removed: {p}")
    else:
        print("best_model.ckpt not found; no folders removed.")


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

    # =============================================================
    # SECTION: Loading Data
    # =============================================================
    df_train, df_test = load_data(hps)
    collate_fn = get_collate_fn(hps)
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

    pool_net, pool_model = setup_model(hps, is_init=args.init)
    # =============================================================
    # SECTION: Setup Logger, Dataloader
    # =============================================================
    fold_metrics = []
    fold_checkpoints = []

    if not args.eval:
        splitter, num_folds = create_data_split(df_train, target_labels, use_kfold=hps.train.use_Kfold)

        for fold, (train_idx, val_idx) in enumerate(splitter):
            logger.info(f"\n{'='*20} Fold {fold+1}/{num_folds} {'='*20}")

            train_fold = df_train.iloc[train_idx].reset_index(drop=True)
            val_fold = df_train.iloc[val_idx].reset_index(drop=True)

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
        best_fold_idx, production_path = select_best_fold(fold_metrics, fold_checkpoints, hps.model_dir, logger)
        save_fold_info(best_fold_idx, fold_metrics, hps.model_dir)
    elif not args.eval:
        best_fold_idx = 0
        best_model_path = fold_checkpoints[best_fold_idx]
        production_path = os.path.join(hps.model_dir, "best_model.ckpt")
        shutil.copy2(best_model_path, production_path)
        save_fold_info(best_fold_idx, fold_metrics, hps.model_dir)
    else:
        # In eval mode, load best_fold_idx from existing info_fold.pkl
        info_fold_data = load_fold_info(hps.model_dir)
        best_fold_idx = info_fold_data.get("best_fold_idx", 0)

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

    runner_lightning = lightning_wrapper.CoughClassificationRunner.load_from_checkpoint(
        os.path.join(hps.model_dir, "best_model.ckpt"),
        model=pool_model,
        hps=hps, custom_logger=logger
    )
    runner_lightning.eval()
    trainer = L.Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu", devices="auto")

    info_fold = load_fold_info(hps.model_dir)
    best_fold_idx = info_fold["best_fold_idx"]
    fold_metrics = info_fold["fold_metrics"]

    ###### Calibrate Threshold or Load if Exist
    if os.path.exists(os.path.join(model_dir, "probs_threshold.pkl")):
        with open(os.path.join(model_dir, "probs_threshold.pkl"), "rb") as f:
            runner_lightning.probs_threshold = pickle.load(f)['probs_threshold']
    else:
        splitter, num_folds = create_data_split(df_train, target_labels, use_kfold=hps.train.use_Kfold)
        train_idx, val_idx = list(splitter)[best_fold_idx]

        train_fold = df_train.iloc[train_idx].reset_index(drop=True)
        val_fold = df_train.iloc[val_idx].reset_index(drop=True)
        train_loader, val_loader = prepare_fold_data(train_fold, val_fold, hps, best_fold_idx, collate_fn)

        runner_lightning.calibrate_threshold = True
        results = trainer.test(runner_lightning, dataloaders=val_loader)[0]
        runner_lightning.calibrate_threshold = False
        
        payload = {
            "probs_threshold": runner_lightning.probs_threshold,
        }
        with open(os.path.join(model_dir, "probs_threshold.pkl"), "wb") as f:
            pickle.dump(payload, f)

    ############################################# Overall Report #########################################################
    with open(f"{model_dir}/result_overall.txt", "w") as f:
        f.write(f"")

    df_train_eval = pd.read_csv(f'/run/media/fourier/Data1/Pras/Thesis_Nexus/NXSCough/data/{hps.data.metadata_csv}.train')
    df_train_eval = df_train_eval.reset_index(drop=True)
    df_train_eval = df_train_eval[hps.data.column_order]

    results_dict = evaluate_on_dataset(runner_lightning, trainer, df_train_eval, hps, best_fold_idx, collate_fn)
    write_results_to_file("result_overall.txt", results_dict, model_dir, {0: "Train"})

    df_test_eval = pd.read_csv(f'/run/media/fourier/Data1/Pras/Thesis_Nexus/NXSCough/data/{hps.data.metadata_csv}.test')
    df_test_eval = df_test_eval.reset_index(drop=True)
    df_test_eval = df_test_eval[hps.data.column_order]

    runner_lightning.generate_figure = True
    results_dict = evaluate_on_dataset(runner_lightning, trainer, df_test_eval, hps, best_fold_idx, collate_fn)
    write_results_to_file("result_overall.txt", results_dict, model_dir, {0: "Test"})
    runner_lightning.generate_figure = False

    df_cirdz = pd.read_csv(f'/run/media/fourier/Data1/Pras/Thesis_Nexus/NXSCough/data/cirdz.csv.test')
    df_cirdz = df_cirdz.reset_index(drop=True)
    df_cirdz = df_cirdz[hps.data.column_order]

    results_dict = evaluate_on_dataset(runner_lightning, trainer, df_cirdz, hps, best_fold_idx, collate_fn)
    write_results_to_file("result_overall.txt", results_dict, model_dir, {0: "Unseen"})
    ############################################# PerDB Report ###########################################################
    # Handle result_summary.txt versioning
    rotate_result_summary(model_dir)
    with open(f"{model_dir}/result_summary.txt", "w") as f:
        f.write(f"{'='*25} Train Phase {'='*25}\n")
    df_train_eval = pd.read_csv(f'/run/media/fourier/Data1/Pras/Thesis_Nexus/NXSCough/data/{hps.data.metadata_csv}.train')
    df_train_eval = df_train_eval.reset_index(drop=True)
    df_train_eval = df_train_eval[hps.data.column_order + ['db']]

    results_dict = evaluate_on_dataset(runner_lightning, trainer, df_train_eval, hps, best_fold_idx, collate_fn, db_column='db')
    write_results_to_file("result_summary.txt", results_dict, model_dir, db_map)

    with open(f"{model_dir}/result_summary.txt", "a") as f:
        f.write(f"\n{'='*25} Test Phase {'='*25}\n")
    df_test_eval = pd.read_csv(f'/run/media/fourier/Data1/Pras/Thesis_Nexus/NXSCough/data/{hps.data.metadata_csv}.test')
    df_test_eval = df_test_eval.reset_index(drop=True)
    df_test_eval = df_test_eval[hps.data.column_order + ['db']]

    results_dict = evaluate_on_dataset(runner_lightning, trainer, df_test_eval, hps, best_fold_idx, collate_fn, db_column='db')
    write_results_to_file("result_summary.txt", results_dict, model_dir, db_map)

    df_cirdz = pd.read_csv(f'/run/media/fourier/Data1/Pras/Thesis_Nexus/NXSCough/data/cirdz.csv.test')
    df_cirdz = df_cirdz.reset_index(drop=True)
    df_cirdz = df_cirdz[hps.data.column_order]

    results_dict = evaluate_on_dataset(runner_lightning, trainer, df_cirdz, hps, best_fold_idx, collate_fn)
    write_results_to_file("result_summary.txt", results_dict, model_dir, {0: "CIRDZ"})

    # =============================================================
    # SECTION: Cleaning
    # =============================================================
    cleanup_fold_directories(hps.model_dir, best_fold_idx)


if __name__ == "__main__":
    main()
