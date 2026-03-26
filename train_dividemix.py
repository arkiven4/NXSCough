"""
DivideMix training for cough / TB detection.
Reference: Li et al. "DivideMix: Learning with Noisy Labels as
           Semi-supervised Learning" (ICLR 2020)

Algorithm:
  Phase 1 – Warmup   : both net1 and net2 trained with CE on all data
                        (Lightning + CoughClassificationRunner, one net at a time)
  Phase 2 – Co-train : alternating manual loop
                         eval_train → 2-component GMM → labeled / unlabeled split
                         → MixMatch update (co-guess + label-refinement + MixUp)
"""

import json
import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

import lightning_wrapper
import utils
import train
from cough_datasets import CoughDatasets

torch.set_float32_matmul_precision("medium")


# =====================================================================
# Loss helpers  (identical semantics to original DivideMix)
# =====================================================================

class SemiLossAudio:
    """Lx = soft cross-entropy on labeled, Lu = MSE on unlabeled guesses."""
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u,
                 epoch, warm_up, lambda_u=25.0):
        probs_u = torch.softmax(outputs_u, dim=1)
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u) ** 2)
        ramp = float(np.clip((epoch - warm_up) / 16.0, 0.0, 1.0))
        return Lx, Lu, lambda_u * ramp


class NegEntropyAudio:
    """Negative entropy — penalises overconfident predictions (asym noise)."""
    def __call__(self, outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log() * probs, dim=1))


# =====================================================================
# DivideMix dataset  (wraps CoughDatasets, three modes)
# =====================================================================

class DivideMixDataset(torch.utils.data.Dataset):
    """
    Modes
    -----
    'all'       → (audio, label, original_idx)         eval / warmup
    'labeled'   → (audio1_aug, audio2_aug, label, prob) clean samples
    'unlabeled' → (audio1_aug, audio2_aug)              noisy samples

    Two augmented views come from calling the underlying dataset twice;
    because augmentation is stochastic each call produces a different view.
    """

    def __init__(self, df, hps, mode='all', pred=None, prob=None, train_mode=True,
                 use_precomputed=False, precomputed_dir=None):
        self.mode = mode
        self.prob = prob

        data = df
        feature_path_col = None
        if use_precomputed and precomputed_dir:
            mapping_train = pd.read_csv(os.path.join(precomputed_dir, "feature_mapping_train.csv"))
            data = df.merge(
                mapping_train[["path_file", "feature_path"]],
                on="path_file",
                how="left",
            )
            feature_path_col = list(data.columns).index("feature_path")

        self._ds = CoughDatasets(
            data.values, hps.data,
            wav_stats_path=f"{hps.model_dir}/wav_stats.pickle" if not use_precomputed else None,
            train=train_mode,
            use_precomputed=use_precomputed,
        )
        if use_precomputed and feature_path_col is not None:
            self._ds.set_feature_path_column(feature_path_col)

        if mode == 'labeled' and pred is not None:
            self.indices = np.where(pred)[0]
        elif mode == 'unlabeled' and pred is not None:
            self.indices = np.where(~pred)[0]
        else:
            self.indices = np.arange(len(df))

    # item layout from CoughDatasets.__getitem__:
    # (orig_idx, name, audio, dse_id[1,C], spk_id, gndr_id, tabular)
    def _label(self, item):
        return torch.argmax(item[3].squeeze(0)).long()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ri = int(self.indices[idx])
        item1 = self._ds[ri]
        audio1 = item1[2]

        if self.mode == 'labeled':
            audio2 = self._ds[ri][2]          # different random aug
            return audio1, audio2, self._label(item1), float(self.prob[ri])

        elif self.mode == 'unlabeled':
            audio2 = self._ds[ri][2]
            return audio1, audio2

        else:  # 'all'
            return audio1, self._label(item1), ri


# =====================================================================
# Collate functions
# =====================================================================

def _pad_batch(audios):
    """[1,C,T] or [1,T] list → [B,C,max_T], mask [B,max_T]."""
    max_len = max(a.shape[-1] for a in audios)
    B = len(audios)
    s = audios[0]
    C = s.shape[1] if s.ndim == 3 else 1
    out = torch.zeros(B, C, max_len, dtype=s.dtype)
    mask = torch.zeros(B, max_len)
    for i, a in enumerate(audios):
        L = a.shape[-1]
        out[i, :, :L] = a if a.ndim == 3 else a.unsqueeze(0)
        mask[i, :L] = 1.0
    return out, mask


def _dm_labeled_collate(batch):
    a1s, a2s, labels, probs = zip(*batch)
    a1, mask = _pad_batch(list(a1s))
    a2, _    = _pad_batch(list(a2s))
    return (a1, a2,
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(probs,  dtype=torch.float32),
            mask)


def _dm_unlabeled_collate(batch):
    a1s, a2s = zip(*batch)
    a1, mask = _pad_batch(list(a1s))
    a2, _    = _pad_batch(list(a2s))
    return a1, a2, mask


def _dm_all_collate(batch):
    audios, labels, idxs = zip(*batch)
    a, mask = _pad_batch(list(audios))
    return (a,
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(idxs,   dtype=torch.long),
            mask)


# =====================================================================
# Core DivideMix functions  (mirror original Train_cifar.py structure)
# =====================================================================

def _make_dm_loaders(df, hps, pred, prob, batch_size, num_workers=8,
                     use_precomputed=False, precomputed_dir=None):
    """Build labeled + unlabeled loaders for one co-training step."""
    lab_ds = DivideMixDataset(
        df, hps, mode='labeled', pred=pred, prob=prob, train_mode=True,
        use_precomputed=use_precomputed, precomputed_dir=precomputed_dir,
    )
    unl_ds = DivideMixDataset(
        df, hps, mode='unlabeled', pred=pred, prob=prob, train_mode=True,
        use_precomputed=use_precomputed, precomputed_dir=precomputed_dir,
    )
    kw = dict(num_workers=num_workers, drop_last=True, pin_memory=True)
    lab_loader = DataLoader(lab_ds, batch_size=batch_size, shuffle=True,
                            collate_fn=_dm_labeled_collate,   **kw)
    unl_loader = DataLoader(unl_ds, batch_size=batch_size, shuffle=True,
                            collate_fn=_dm_unlabeled_collate, **kw)
    return lab_loader, unl_loader


def eval_train(net, eval_loader, all_loss, device, n_samples,
               class0_mode='default', log_fn=None):
    """
    Compute per-sample loss → normalise → class-conditional GMM → clean probability.

    class0_mode controls how class-0 (negative / non-TB) samples are treated:
      'default' – GMM applied to class 0 as well; noisy class-0 samples go to
                  the unlabeled set, same as class 1  (original DivideMix behaviour)
      'keep'   – skip GMM for class 0; all negatives are labelled clean (prob=1)
                 (use when you trust negative labels and only want to clean positives)

    Why class-conditional GMM?
    A single GMM on all losses fails on imbalanced data: the minority class (TB)
    is inherently harder, so it always sits in the high-loss component and gets
    labelled 'noisy' regardless of label quality — causing progressive AUROC
    collapse.  Per-class GMMs separate clean/noisy *within* each class.
    """
    net.eval()
    net.to(device)
    losses     = torch.zeros(n_samples)
    labels_all = torch.zeros(n_samples, dtype=torch.long)

    with torch.no_grad():
        for audio, labels, indices, masks in eval_loader:
            audio, labels, masks = audio.to(device), labels.to(device), masks.to(device)
            logits = net(audio, attention_mask=masks)['disease_logits']
            if logits.shape[-1] == 1:
                loss = F.binary_cross_entropy_with_logits(
                    logits.squeeze(-1), labels.float(), reduction='none')
            else:
                loss = F.cross_entropy(logits, labels, reduction='none')
            for b, orig_idx in enumerate(indices):
                losses[orig_idx.item()]     = loss[b].item()
                labels_all[orig_idx.item()] = labels[b].item()

    losses = (losses - losses.min()) / (losses.max() - losses.min() + 1e-8)
    all_loss.append(losses)

    input_loss    = torch.stack(all_loss[-5:]).mean(0) if len(all_loss) >= 5 else losses
    input_loss_np = input_loss.numpy()
    labels_np     = labels_all.numpy()

    prob = np.zeros(n_samples)
    for cls in np.unique(labels_np):
        mask = labels_np == cls

        if cls == 0 and class0_mode == 'keep':
            prob[mask] = 1.0   # all class-0 samples are considered clean
            if log_fn:
                log_fn(f"    GMM cls=0  kept as clean (class0_mode='keep')  n={mask.sum()}")
            continue

        if mask.sum() < 4:
            prob[mask] = 1.0
            continue

        cls_loss = input_loss_np[mask].reshape(-1, 1)
        gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
        gmm.fit(cls_loss)
        p = gmm.predict_proba(cls_loss)
        prob[mask] = p[:, gmm.means_.argmin()]   # low-loss component = clean
        if log_fn:
            log_fn(f"    GMM cls={cls}  n={mask.sum()}  "
                   f"means={gmm.means_.flatten().round(4)}  "
                   f"weights={gmm.weights_.round(3)}")

    return prob, all_loss


def _probs_from_logits(logits):
    """
    Convert model logits to class probabilities.
    - binary head [B,1]   -> [B,2] via sigmoid
    - multiclass [B,C]    -> [B,C] via softmax
    """
    if logits.shape[-1] == 1:
        p1 = torch.sigmoid(logits)
        return torch.cat([1.0 - p1, p1], dim=1)
    return F.softmax(logits, dim=1)


def _to_multiclass_logits(logits):
    """
    Make logits compatible with softmax-based multi-class losses.
    - binary head [B,1]   -> [B,2] as [0, z]
    - multiclass [B,C]    -> [B,C]
    """
    if logits.shape[-1] == 1:
        return torch.cat([torch.zeros_like(logits), logits], dim=1)
    return logits


def train_dividemix(epoch, net, peer_net, optimizer,
                    lab_loader, unl_loader,
                    warm_up, device, num_class, alpha, lambda_u, T,
                    noise_mode='sym', log_fn=None):
    """
    One MixMatch co-training epoch.
    Mirrors train() in Train_cifar.py:
      - co-guess soft labels for unlabeled (both nets × 2 views)
      - refine labels for labeled (clean-prob weighted)
      - MixUp → SemiLoss + entropy penalty
    net trains; peer_net is fixed.
    """
    net.train()
    peer_net.eval()

    criterion   = SemiLossAudio()
    neg_entropy = NegEntropyAudio()
    unl_iter    = iter(unl_loader)
    num_iter    = len(lab_loader)

    for batch_idx, (x1, x2, labels_x, w_x, masks_x) in enumerate(lab_loader):
        try:
            u1, u2, masks_u = next(unl_iter)
        except StopIteration:
            unl_iter = iter(unl_loader)
            u1, u2, masks_u = next(unl_iter)

        B = x1.size(0)
        x1, x2   = x1.to(device),      x2.to(device)
        u1, u2   = u1.to(device),      u2.to(device)
        labels_x = labels_x.to(device)
        w_x      = w_x.to(device).float().view(-1, 1)
        masks_x  = masks_x.to(device)
        masks_u  = masks_u.to(device)

        labels_oh = torch.zeros(B, num_class, device=device)
        labels_oh.scatter_(1, labels_x.view(-1, 1), 1)

        with torch.no_grad():
            # --- co-guess labels for unlabeled (avg of both nets on 2 views) ---
            ou11 = peer_net(u1, attention_mask=masks_u)['disease_logits']
            ou12 = peer_net(u2, attention_mask=masks_u)['disease_logits']
            ou21 = net(u1,     attention_mask=masks_u)['disease_logits']
            ou22 = net(u2,     attention_mask=masks_u)['disease_logits']
            pu   = (_probs_from_logits(ou11) + _probs_from_logits(ou12) +
                _probs_from_logits(ou21) + _probs_from_logits(ou22)) / 4
            ptu      = pu ** (1.0 / T)
            targets_u = (ptu / ptu.sum(dim=1, keepdim=True)).detach()

            # --- label refinement for labeled ---
            ox1 = net(x1, attention_mask=masks_x)['disease_logits']
            ox2 = net(x2, attention_mask=masks_x)['disease_logits']
            px  = (_probs_from_logits(ox1) + _probs_from_logits(ox2)) / 2
            px  = w_x * labels_oh + (1.0 - w_x) * px
            ptx      = px ** (1.0 / T)
            targets_x = (ptx / ptx.sum(dim=1, keepdim=True)).detach()

        # --- MixUp ---
        lam = np.random.beta(alpha, alpha)
        lam = max(lam, 1.0 - lam)

        all_in  = torch.cat([x1, x2, u1, u2], dim=0)
        all_tgt = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)
        all_msk = torch.cat([masks_x, masks_x, masks_u, masks_u], dim=0)

        perm      = torch.randperm(all_in.size(0), device=device)
        mixed_in  = lam * all_in  + (1.0 - lam) * all_in[perm]
        mixed_tgt = lam * all_tgt + (1.0 - lam) * all_tgt[perm]
        mixed_msk = torch.max(all_msk, all_msk[perm])

        logits_all_raw = net(mixed_in, attention_mask=mixed_msk, train=True)['disease_logits']
        logits_all = _to_multiclass_logits(logits_all_raw)
        logits_x_  = logits_all[:B * 2]
        logits_u_  = logits_all[B * 2:]

        Lx, Lu, lamb = criterion(
            logits_x_, mixed_tgt[:B * 2],
            logits_u_, mixed_tgt[B * 2:],
            epoch + batch_idx / num_iter, warm_up, lambda_u,
        )

        # Use class frequencies from the labeled batch as prior.
        # Original DivideMix uses uniform prior (fine for balanced CIFAR).
        # For imbalanced data (e.g. 80% non-TB / 20% TB), a uniform prior
        # pushes predictions toward 50/50, collapsing AUROC over epochs.
        prior    = labels_oh.mean(0).detach()
        prior    = (prior + 1e-8) / (prior.sum() + 1e-8)
        pred_avg = F.softmax(logits_all, dim=1).mean(0)
        penalty  = torch.sum(prior * torch.log(prior / (pred_avg + 1e-8)))
        if noise_mode == 'asym':
            penalty = penalty + neg_entropy(logits_all)

        loss = Lx + lamb * Lu + penalty
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if log_fn and batch_idx % 50 == 0:
            log_fn(f"    [{batch_idx:4d}/{num_iter}] "
                   f"Lx={Lx.item():.3f}  Lu={Lu.item():.3f}  lamb={lamb:.3f}")


def eval_agreement(net1, net2, eval_loader, device, log_fn=None):
    """
    Compute net1–net2 prediction agreement on the training set.

    Used as a proxy convergence metric when test labels are unreliable.
    Higher agreement → more stable co-training; watch for it plateauing or dropping.

    eval_loader must use _dm_all_collate format: (audio, labels, indices, masks)
    Returns agreement rate in [0, 1].
    """
    net1.eval()
    net2.eval()
    agree = total = 0

    with torch.no_grad():
        for audio, labels, _, masks in eval_loader:
            audio, masks = audio.to(device), masks.to(device)
            out1 = net1(audio, attention_mask=masks)['disease_logits']
            out2 = net2(audio, attention_mask=masks)['disease_logits']

            if out1.shape[-1] == 1:
                pred1 = (torch.sigmoid(out1.squeeze(-1)) > 0.5).long()
                pred2 = (torch.sigmoid(out2.squeeze(-1)) > 0.5).long()
            else:
                pred1 = out1.argmax(dim=1)
                pred2 = out2.argmax(dim=1)

            agree += (pred1 == pred2).sum().item()
            total += labels.size(0)

    rate = agree / max(total, 1)
    if log_fn:
        log_fn(f"  Net1–Net2 agreement: {agree}/{total} = {rate:.4f}")
    return rate


def save_label_analysis(net1, net2, eval_loader, noisy_labels,
                        prob1, prob2, p_threshold, save_path, device, log_fn=None):
    """
    Run ensemble inference on the full training set and write per-sample analysis to CSV.

    Columns
    -------
    noisy_label     : original label from the dataset (may be wrong)
    clean_prob_net1 : GMM cleanness probability from net1  (↑ = more likely clean label)
    clean_prob_net2 : GMM cleanness probability from net2
    clean_prob_avg  : average of the two — recommended label-quality score
    is_clean        : True when clean_prob_avg > p_threshold
    pred_prob_tb    : ensemble P(TB) from the final net1+net2 models
    pred_label      : hard predicted label (1=TB, 0=NTB) at threshold 0.5

    How to read the output
    ----------------------
    - pred_label  : the denoised label suggested by the trained ensemble
    - clean_prob_avg close to 1 + pred_label matches noisy_label → high-confidence clean sample
    - clean_prob_avg close to 0 → GMM flagged as noisy; pred_label may differ from noisy_label
    """
    net1.eval()
    net2.eval()
    n = len(noisy_labels)
    pred_probs = torch.zeros(n)

    with torch.no_grad():
        for audio, _, indices, masks in eval_loader:
            audio, masks = audio.to(device), masks.to(device)
            out1 = net1(audio, attention_mask=masks)['disease_logits']
            out2 = net2(audio, attention_mask=masks)['disease_logits']

            if out1.shape[-1] == 1:
                p1 = torch.sigmoid(out1.squeeze(-1))
                p2 = torch.sigmoid(out2.squeeze(-1))
            else:
                p1 = F.softmax(out1, dim=1)[:, 1]
                p2 = F.softmax(out2, dim=1)[:, 1]

            for b, idx in enumerate(indices):
                pred_probs[idx.item()] = ((p1[b] + p2[b]) / 2).item()

    clean_avg = (prob1 + prob2) / 2
    pred_np   = pred_probs.numpy()

    result = pd.DataFrame({
        'noisy_label':     noisy_labels,
        'clean_prob_net1': prob1,
        'clean_prob_net2': prob2,
        'clean_prob_avg':  clean_avg,
        'is_clean':        clean_avg > p_threshold,
        'pred_prob_tb':    pred_np,
        'pred_label':      (pred_np > 0.5).astype(int),
    })
    result.to_csv(save_path, index=False)

    if log_fn:
        n_clean    = int(result['is_clean'].sum())
        n_pred_tb  = int(result['pred_label'].sum())
        n_noisy_tb = int((noisy_labels == 1).sum())
        n_changed  = int((result['pred_label'].values != noisy_labels).sum())
        log_fn(f"  Label analysis saved → {save_path}")
        log_fn(f"  GMM clean samples : {n_clean} / {n}")
        log_fn(f"  Pred TB (class 1) : {n_pred_tb}  (noisy TB: {n_noisy_tb})")
        log_fn(f"  Label changes     : {n_changed} samples differ from noisy_label")


def test_dividemix(net1, net2, test_loader, device, log_fn=None):
    """
    Ensemble of net1+net2.  Mirrors test() in Train_cifar.py.
    test_loader uses CoughDatasetsCollate format:
      (wav_names, wav_padded, None, attention_masks, dse_ids, [...])
    Returns (labels_np, probs_np).
    """
    net1.eval()
    net2.eval()
    all_labels, all_probs, all_logits1, all_logits2 = [], [], [], []

    with torch.no_grad():
        for _, audio, _, masks, dse_ids, _ in test_loader:
            audio, masks = audio.to(device), masks.to(device)
            out1 = net1(audio, attention_mask=masks)['disease_logits']
            out2 = net2(audio, attention_mask=masks)['disease_logits']
            
            if out1.shape[-1] == 1:
                p1 = torch.sigmoid(out1.squeeze(-1))
                p2 = torch.sigmoid(out2.squeeze(-1))
            else:
                p1 = F.softmax(out1, dim=1)[:, 1]
                p2 = F.softmax(out2, dim=1)[:, 1]
            
            probs  = (p1 + p2) / 2
            labels = torch.argmax(dse_ids.float(), dim=1)
            
            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())
            all_logits1.append(out1.cpu())
            all_logits2.append(out2.cpu())

    labels_np = torch.cat(all_labels).numpy()
    probs_np = torch.cat(all_probs).numpy()
    logits1_np = torch.cat(all_logits1).numpy()
    logits2_np = torch.cat(all_logits2).numpy()
    
    if log_fn:
        log_fn(f"    Test labels dist: {np.bincount(labels_np.astype(int))}")
        log_fn(f"    Test probs: min={probs_np.min():.4f}, max={probs_np.max():.4f}, mean={probs_np.mean():.4f}")
        log_fn(f"    Logits1 range: [{logits1_np.min():.4f}, {logits1_np.max():.4f}]")
        log_fn(f"    Logits2 range: [{logits2_np.min():.4f}, {logits2_np.max():.4f}]")
        # Flip diagnostic: mean P(class=1) should be HIGHER for true positives (class=1).
        # If it is lower, predictions are inverted — class index ordering may be wrong.
        for cls in np.unique(labels_np.astype(int)):
            m = labels_np == cls
            log_fn(f"    mean P(class=1) | true_label={cls}: {probs_np[m].mean():.4f}")

    return labels_np, probs_np


# =====================================================================
# Main
# =====================================================================

def main(cli_args=None):
    parser = train.parse_args()
    args   = parser.parse_args(cli_args)

    model_dir = os.path.join("./logs", args.model_name)
    os.makedirs(model_dir, exist_ok=True)

    config_path = args.config_path if args.init else os.path.join(model_dir, "config.json")
    hps = train.load_config(config_path, model_dir, args)
    hps.model.spk_dim = 0

    # --- DivideMix hyper-parameters ---
    dm_config = {
        # Warmup epochs: standard CE on all data before co-training starts.
        # ↑ more  → cleaner network initialisation, slower start
        # ↓ fewer → faster, but noisier starting point for GMM
        "warm_up": 10,

        # Total training epochs (warmup + co-training).
        # ↑ more  → longer convergence, risk of memorising noise if GMM is imperfect
        # ↓ fewer → faster, may underfit
        "num_epochs": 300,

        # MixUp interpolation strength: samples drawn from Beta(alpha, alpha).
        # ↑ larger (e.g. 8) → stronger mixing, more regularisation, smoother boundaries
        # ↓ smaller (→0)    → less mixing, closer to standard training
        "alpha": 4.0,

        # Weight for the unsupervised (unlabeled) loss component.
        # ↑ larger → model relies more on co-guessed pseudo-labels
        # ↓ smaller → model focuses on the labeled (clean) set only
        # Ramped linearly from 0 to lambda_u over 16 epochs after warm_up.
        "lambda_u": 25.0,

        # GMM clean-probability threshold: samples with prob > threshold → labeled set.
        # ↑ larger (e.g. 0.7) → stricter, fewer but purer labeled samples
        # ↓ smaller (e.g. 0.3) → more samples labeled clean, higher noise tolerance
        # 0.5 is the natural midpoint with class-conditional GMM (recommended).
        "p_threshold": 0.7,

        # Temperature for label sharpening (0 < T ≤ 1).
        # ↑ larger (→1) → softer pseudo-labels, more entropy kept
        # ↓ smaller (→0) → harder pseudo-labels, more confident but noisier
        "T": 0.5,

        # Noise model assumption, affects the entropy penalty term:
        # 'sym'  → symmetric noise (random flips); no extra penalty
        # 'asym' → asymmetric noise (class-specific flips); adds neg-entropy penalty
        "noise_mode": "sym",

        "num_class": 2,

        # How class-0 (non-TB / negative) samples are treated in the GMM step:
        # 'default' → GMM applied to class 0 too (original DivideMix behaviour)
        # 'keep'    → skip GMM for class-0; all negatives treated as labeled-clean
        "class0_mode": "keep",
    }
    with open(os.path.join(model_dir, "dm_config.json"), "w") as f:
        json.dump(dm_config, f, indent=2)

    # --- Data ---
    df_train, _ = train.load_data(hps)
    collate_fn        = train.get_collate_fn(hps)
    target_labels     = df_train[hps.data.target_column]

    if not args.use_precomputed:
        utils.compute_spectrogram_stats_from_dataset(
            df_train, hps.data,
            pickle_path=f"{hps.model_dir}/wav_stats.pickle",
        )

    logger_py = utils.get_logger(hps.model_dir)
    logger_py.info(hps)
    logger_py.info(f"DivideMix config: {dm_config}")

    # Validation holdout (warmup early-stopping only)
    sgkf = StratifiedGroupKFold(n_splits=5, random_state=42, shuffle=True)
    wu_trn_idx, wu_val_idx = next(
        sgkf.split(df_train, target_labels, df_train["participant"])
    )
    df_wu_trn = df_train.iloc[wu_trn_idx].reset_index(drop=True)
    df_wu_val = df_train.iloc[wu_val_idx].reset_index(drop=True)
    # Note: no trusted test set — evaluation uses net1–net2 agreement as proxy metric.

    # --- Models ---
    pool_net = train.setup_model(hps, is_init=args.init)
    net1 = pool_net(**hps.model)
    net2 = pool_net(**hps.model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net1, net2 = net1.to(device), net2.to(device)

    opt1 = torch.optim.AdamW(net1.parameters(), lr=hps.train.learning_rate)
    opt2 = torch.optim.AdamW(net2.parameters(), lr=hps.train.learning_rate)

    # =========================================================
    # Phase 1 – Warmup  (Lightning, one net at a time)
    # Uses CoughClassificationRunner + prepare_fold_data as-is.
    # =========================================================
    logger_py.info(f"\n{'='*20} Warmup ({dm_config['warm_up']} epochs) {'='*20}")

    wu_train_loader, wu_val_loader = train.prepare_fold_data(
        df_wu_trn, df_wu_val, hps, collate_fn,
        use_precomputed=args.use_precomputed,
        precomputed_dir=args.precomputed_dir,
    )
    tb_logger = TensorBoardLogger(hps.model_dir, name="dividemix")

    for net, tag in [(net1, "net1"), (net2, "net2")]:
        logger_py.info(f"  Warmup {tag}")
        runner = lightning_wrapper.CoughClassificationRunner(net, hps, logger_py)
        ckpt_cb   = ModelCheckpoint(
            dirpath=f"{hps.model_dir}/warmup_{tag}",
            monitor="val/loss", filename="best", save_top_k=1, mode="min")
        early_stop = EarlyStopping(monitor="val/loss", patience=5, mode="min")
        trainer = L.Trainer(
            max_epochs=dm_config["warm_up"],
            callbacks=[ckpt_cb, early_stop],
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices="auto",
            default_root_dir=hps.model_dir,
            logger=tb_logger,
        )
        trainer.fit(runner,
                    train_dataloaders=wu_train_loader,
                    val_dataloaders=wu_val_loader)
        # net weights updated in-place (runner.model IS net via shared reference)

    # =========================================================
    # Phase 2 – Co-training  (manual loop, matches original)
    # =========================================================
    logger_py.info(f"\n{'='*20} Co-training {'='*20}")

    eval_ds = DivideMixDataset(
        df_train, hps, mode='all', train_mode=False,
        use_precomputed=args.use_precomputed, precomputed_dir=args.precomputed_dir,
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=hps.train.batch_size, shuffle=False,
        num_workers=8, collate_fn=_dm_all_collate, pin_memory=True,
    )

    all_loss       = [[], []]
    n_samples      = len(df_train)
    metrics_history = {}

    for epoch in range(dm_config["warm_up"], dm_config["num_epochs"] + 1):
        logger_py.info(f"\n{'='*20} Epoch {epoch}/{dm_config['num_epochs']} {'='*20}")

        # --- GMM evaluation ---
        c0_mode = dm_config["class0_mode"]
        prob1, all_loss[0] = eval_train(
            net1, eval_loader, all_loss[0], device, n_samples,
            class0_mode=c0_mode, log_fn=logger_py.info)
        prob2, all_loss[1] = eval_train(
            net2, eval_loader, all_loss[1], device, n_samples,
            class0_mode=c0_mode, log_fn=logger_py.info)

        pred1 = prob1 > dm_config["p_threshold"]
        pred2 = prob2 > dm_config["p_threshold"]
        logger_py.info(
            f"  Clean — net1:{pred1.sum()}  net2:{pred2.sum()}  total:{n_samples}")

        # --- Train net1 with net2's co-divide ---
        if pred2.sum() > 0 and (~pred2).sum() > 0:
            lab_loader, unl_loader = _make_dm_loaders(
                df_train, hps, pred2, prob2, hps.train.batch_size,
                use_precomputed=args.use_precomputed,
                precomputed_dir=args.precomputed_dir,
            )
            train_dividemix(
                epoch, net1, net2, opt1, lab_loader, unl_loader,
                dm_config["warm_up"], device,
                dm_config["num_class"], dm_config["alpha"],
                dm_config["lambda_u"], dm_config["T"], dm_config["noise_mode"],
                log_fn=logger_py.info,
            )
        else:
            logger_py.warning("  net2 GMM degenerate — skipping net1 update")

        # --- Train net2 with net1's co-divide ---
        if pred1.sum() > 0 and (~pred1).sum() > 0:
            lab_loader, unl_loader = _make_dm_loaders(
                df_train, hps, pred1, prob1, hps.train.batch_size,
                use_precomputed=args.use_precomputed,
                precomputed_dir=args.precomputed_dir,
            )
            train_dividemix(
                epoch, net2, net1, opt2, lab_loader, unl_loader,
                dm_config["warm_up"], device,
                dm_config["num_class"], dm_config["alpha"],
                dm_config["lambda_u"], dm_config["T"], dm_config["noise_mode"],
                log_fn=logger_py.info,
            )
        else:
            logger_py.warning("  net1 GMM degenerate — skipping net2 update")

        # --- Proxy metrics (no trusted test set) ---
        # Net1–Net2 agreement: proxy for training stability.
        #   Rising   → networks converging on consistent predictions (good)
        #   Dropping → co-training destabilising (check lambda_u / p_threshold)
        # Clean fraction: fraction of training samples GMM considers clean.
        #   Too low  → most data pushed to unlabeled; semi-supervised signal dominates
        #   Too high → GMM not filtering effectively; increase p_threshold
        agreement = eval_agreement(net1, net2, eval_loader, device, log_fn=logger_py.info)
        clean_frac = ((pred1.sum() + pred2.sum()) / 2) / n_samples
        logger_py.info(f"  Clean fraction (avg): {clean_frac:.3f}")

        epoch_metrics = {
            "epoch":        epoch,
            "agreement":    agreement,
            "clean_frac":   float(clean_frac),
            "n_clean_net1": int(pred1.sum()),
            "n_clean_net2": int(pred2.sum()),
        }

        metrics_history[epoch] = epoch_metrics

        if epoch % 10 == 0:
            torch.save(net1.state_dict(), f"{hps.model_dir}/net1_ep{epoch:04d}.pt")
            torch.save(net2.state_dict(), f"{hps.model_dir}/net2_ep{epoch:04d}.pt")

    torch.save(net1.state_dict(), f"{hps.model_dir}/net1_final.pt")
    torch.save(net2.state_dict(), f"{hps.model_dir}/net2_final.pt")
    with open(os.path.join(hps.model_dir, "dm_metrics.pkl"), "wb") as f:
        pickle.dump(metrics_history, f)

    # --- Save label analysis (denoised labels + cleanness scores) ---
    logger_py.info("\nGenerating label analysis...")
    save_label_analysis(
        net1, net2, eval_loader,
        noisy_labels=df_train[hps.data.target_column].values,
        prob1=prob1, prob2=prob2,
        p_threshold=dm_config["p_threshold"],
        save_path=os.path.join(hps.model_dir, "label_analysis.csv"),
        device=device, log_fn=logger_py.info,
    )

    logger_py.info("DivideMix training complete.")


if __name__ == "__main__":
    main()
