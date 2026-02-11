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
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split, StratifiedGroupKFold
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
def main(cli_args=None):
    parser = train.parse_args()
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
    N_RUNS = 100
    N_FOLDS = 5
    TAU = 0.01
    RANDOM_STATE = 1
    MEMORY_NOISE_THRES = 0.5
    NOISY_DROP = 0.8 # Dropping bottom N * NOISY_DROP of the dataset 
    Objective_Function_Weights = [0.5, 0.0, 0.5]
    Youden_Weights = [1.3, 0.7]

    kf = KFold(n_splits=N_FOLDS, random_state=RANDOM_STATE, shuffle=True)
    fold_splits = list(kf.split(df_train, target_labels))

    monitor_metrics = {
        'N_RUNS': N_RUNS,
        'N_FOLDS': N_FOLDS,
        'TAU': TAU,
        'MEMORY_NOISE_THRES': MEMORY_NOISE_THRES,
        'NOISY_DROP': NOISY_DROP,
        'Objective_Function_Weights': Objective_Function_Weights,
        'Youden_Weights': Youden_Weights,
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
            train_loader, val_loader = train.prepare_fold_data(
                train_fold, val_fold, hps, fold, collate_fn,
                use_precomputed=args.use_precomputed,
                precomputed_dir=args.precomputed_dir
            )

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

        metrics = ['test_accs', 'test_sens', 'test_spec', 'test_youden', 'confidence', ]
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
