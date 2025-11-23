import argparse, inspect, json, os, pickle, socket, subprocess, warnings, random, math, librosa, shutil
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning as L
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

import commons, models, utils
from cough_datasets import CoughDatasets, CoughDatasetsCollate

torch.set_float32_matmul_precision("medium")

def compute_class_weights(df, target_col, device="cuda"):
    disease_codes = df[target_col].unique().tolist()
    class_frequencies = df[target_col].value_counts().to_dict()
    total_samples = len(df)

    class_weights = {
        cls: total_samples / (len(disease_codes) * freq) if freq != 0 else 0
        for cls, freq in class_frequencies.items()
    }

    weights_list = [class_weights[cls] for cls in disease_codes]
    return torch.tensor(weights_list, device=device, dtype=torch.float)


def compute_wav_stats(df, path_col, pickle_path="wav_stats.pickle"):
    if os.path.exists(pickle_path):
        with open(pickle_path, "rb") as f:
            return pickle.load(f)

    means, stds = [], []
    paths = df[path_col].dropna().tolist()

    for path in tqdm(paths, desc="Processing WAV files", unit="file"):
        if not os.path.isfile(path):
            continue
        try:
            audio, _ = librosa.load(path, sr=None, mono=True)
            means.append(np.mean(audio))
            stds.append(np.std(audio))
        except Exception:
            continue

    stats = {
        "mean_db": float(np.mean(means)),
        "std_db": float(np.mean(stds))
    }

    with open(pickle_path, "wb") as f:
        pickle.dump(stats, f)

    return stats


#######################################################################
parser = argparse.ArgumentParser()
parser.add_argument("--init", action="store_true")
parser.add_argument("--model_name", type=str, default="try_wavlmlora_downstream")
parser.add_argument("--config_path", type=str, default="configs/ssl_finetuning.json")
args = parser.parse_args()

model_dir = os.path.join("./logs", args.model_name)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

port = utils.get_free_port()
subprocess.Popen(
    ["tensorboard", "--logdir", model_dir, "--port", str(port), "--host", "0.0.0.0"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)

config_save_path = os.path.join(model_dir, "config.json")
if args.init:
    with open(args.config_path, "r") as f:
        data = f.read()
    with open(config_save_path, "w") as f:
        f.write(data)
else:
    with open(config_save_path, "r") as f:
        data = f.read()
config = json.loads(data)

hps = utils.HParams(**config)
hps.model_dir = model_dir

# =============================================================
# SECTION: Loading Data
# =============================================================
df_train = pd.read_csv(f'/run/media/fourier/Data1/Pras/Thesis_Nexus/NXSCough/data/{hps.data.metadata_csv}.train')
df_test = pd.read_csv(f'/run/media/fourier/Data1/Pras/Thesis_Nexus/NXSCough/data/{hps.data.metadata_csv}.test')

df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

df_train = df_train[hps.data.column_order]
df_test = df_test[hps.data.column_order]

# class_weights_tensor = compute_class_weights(df_train, hps.data.target_column)
# compute_wav_stats(df_train, "path_file", pickle_path="wav_stats.pickle")

collate_fn = CoughDatasetsCollate(hps.data.many_class)
# =============================================================
# SECTION: Model Setup
# =============================================================
logger = utils.get_logger(hps.model_dir)
logger.info(hps)

logger.info(f"======================================")
logger.info(f"✨ Loss: {hps.train.loss_function}")
logger.info(f"✨ Use Between Class Training: {hps.data.mix_audio}")
logger.info(f"✨ Use Augment: {hps.data.augment_data}")
logger.info(f"✨ Use Rawboost Augment: {hps.data.augment_rawboost}")
logger.info(f"✨ Padding Type: {hps.data.pad_types}")
logger.info(f"✨ Using Model: {hps.model.pooling_model}")
logger.info(f"✨ Tensorboard: http://100.101.198.75:{port}/#scalars&_smoothingWeight=0")
logger.info(f"======================================")

hps.model.spk_dim = 0
pool_net = getattr(models, hps.model.pooling_model)
pool_model = pool_net(hps.model.feature_dim, **hps.model)
shutil.copy2('./models.py', f'{hps.model_dir}/model_net.py.bak')

# =============================================================
# SECTION: Loop Setup
# =============================================================

class CoughDetectionRunner(L.LightningModule):
    def __init__(self, model, hps, class_weights):
        super().__init__()
        self.model = model
        self.hps = hps
        self.class_weights = class_weights

    def forward(self, x, attention_mask=None):
        x = self.model(x, attention_mask=attention_mask)
        return x
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hps.train.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=2)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "loss/epoch",
            },
        }

    def on_after_backward(self):
        norm_type = 2
        total_norm = 0
        parameters = list(filter(lambda p: p.grad is not None, self.model.parameters()))
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)
        self.log("train/grad_norm", total_norm, sync_dist=True)

    def training_step(self, batch, batch_idx):
        _, audio, attention_masks, dse_ids, othr_ids = batch
        
        out_model = self.forward(audio, attention_mask=attention_masks)
        ld = utils.many_loss_category(out_model["disease_logits"], dse_ids, 
                                      loss_type=self.hps.train.loss_function, weights=self.class_weights)
        loss = sum(ld)
        
        self.log("train/loss_step", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.logger.experiment.add_scalars('loss', {'train': loss}, self.global_step)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        lr = self.optimizers().param_groups[0]["lr"]
        self.log("train/lr", lr, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        _, audio, attention_masks, dse_ids, othr_ids = batch
        
        out_model = self.forward(audio, attention_mask=attention_masks)
        ld = utils.many_loss_category(out_model["disease_logits"], dse_ids, 
                                      loss_type=self.hps.train.loss_function, weights=self.class_weights)

        loss = sum(ld)
        self.logger.experiment.add_scalars('loss', {'valid': loss}, self.global_step)

    def on_test_epoch_start(self):
        self.test_preds = []
        self.test_labels = []

    def test_step(self, batch, batch_idx):
        _, audio, attention_masks, dse_ids, _ = batch

        audio = audio.float().squeeze(1)
        attention_masks = attention_masks.float()
        dse_ids = dse_ids.float()

        logits = self(audio, attention_mask=attention_masks)["disease_logits"]
        preds = torch.argmax(logits, dim=1)
        labels = torch.argmax(dse_ids, dim=1)

        self.test_preds.append(preds.cpu())
        self.test_labels.append(labels.cpu())

    def on_test_epoch_end(self):
        preds = torch.cat(self.test_preds).numpy()
        labels = torch.cat(self.test_labels).numpy()

        cm = confusion_matrix(labels, preds)
        n_classes = cm.shape[0]
        class_labels = [f"Class {i+1}" for i in range(n_classes)]

        acc = accuracy_score(labels, preds)
        b_acc = balanced_accuracy_score(labels, preds)

        sens = np.mean([
            cm[i, i] / cm[i, :].sum()
            for i in range(n_classes) if cm[i, :].sum() > 0
        ])

        spec = np.mean([
            (cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]) /
            (cm.sum() - cm[i, :].sum())
            for i in range(n_classes) if (cm.sum() - cm[i, :].sum()) > 0
        ])

        # Log metrics
        self.log("test_acc", acc, sync_dist=True)
        self.log("test_bacc", b_acc, sync_dist=True)
        self.log("test_sens", sens, sync_dist=True)
        self.log("test_spec", spec, sync_dist=True)

        # Export confusion matrix if needed
        if hasattr(self.hparams, "model_dir"):
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=class_labels, yticklabels=class_labels)
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title("Confusion Matrix")
            plt.savefig(f"{self.hparams.model_dir}/result_cm.png")
            plt.close()

        return {
            "acc": acc,
            "bacc": b_acc,
            "sens": sens,
            "spec": spec,
        }
    
# =============================================================
# SECTION: Setup Logger, Dataloader
# =============================================================
fold_metrics = []
fold_checkpoints = []

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
target_labels = df_train[hps.data.target_column]

for fold, (train_idx, val_idx) in enumerate(skf.split(df_train, target_labels)):
    logger.info(f"\n{'='*20} Fold {fold+1}/{skf.get_n_splits()} {'='*20}")
    
    train_fold = df_train.iloc[train_idx].reset_index(drop=True)
    val_fold = df_train.iloc[val_idx].reset_index(drop=True)
    
    class_weights_tensor = compute_class_weights(train_fold, hps.data.target_column)
    compute_wav_stats(train_fold, "path_file", pickle_path=f"{hps.model_dir}/wav_stats_fold_{fold}.pickle")

    train_dataset = CoughDatasets(train_fold.values, hps.data, wav_stats_path=f"{hps.model_dir}/wav_stats_fold_{fold}.pickle", train=True)
    val_dataset = CoughDatasets(val_fold.values, hps.data, wav_stats_path=f"{hps.model_dir}/wav_stats_fold_{fold}.pickle", train=False)

    train_loader = DataLoader(train_dataset, num_workers=28, shuffle=True, batch_size=hps.train.batch_size,
                              pin_memory=True, drop_last=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, num_workers=28, shuffle=False, batch_size=hps.train.batch_size,
                            pin_memory=True, drop_last=True, collate_fn=collate_fn)

    # Initialize a FRESH model for each fold
    hps.model.spk_dim = 0
    pool_model = pool_net(hps.model.feature_dim, **hps.model)

    checkpoint_callback = ModelCheckpoint(
       dirpath=f"{hps.model_dir}/fold_{fold}",
       monitor="loss/epoch",
       filename=f"pool_fold{fold}_{{epoch:02d}}",
       save_top_k=1,
       mode="min",
    )

    tb_logger = TensorBoardLogger(save_dir=hps.model_dir, name=f"fold_{fold}", sub_dir="train")
    early_stopping = EarlyStopping(monitor="loss/epoch", patience=5, mode="min", verbose=False)

    runner_lightning = CoughDetectionRunner(pool_model, hps=hps, class_weights=class_weights_tensor)
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