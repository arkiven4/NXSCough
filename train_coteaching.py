"""
Co-Teaching training for cough / TB detection.
Reference: Han et al. "Co-teaching: Robust Training of Deep Neural Networks
           with Extremely Noisy Labels" (NeurIPS 2018)

Algorithm:
  Each mini-batch:
    1. Both networks compute per-sample CE losses on the same batch.
    2. Network 1 selects floor((1 - R(t)) * N) small-loss samples  → trains Network 2.
    3. Network 2 selects floor((1 - R(t)) * N) small-loss samples  → trains Network 1.

  Noise rate schedule (after warmup):
    R(t) = min(noise_rate * (epoch - warmup) / T_k,  noise_rate)
    keep_ratio(t) = 1 - R(t)
    epoch = warmup     → keep 100 % (standard CE)
    epoch = warmup+T_k → keep (1 - noise_rate) fraction
    epoch > warmup+T_k → constant at (1 - noise_rate)

Key difference from DivideMix:
  - No GMM, no semi-supervised MixMatch, no separate labeled/unlabeled split.
  - Simpler: only small-loss selection + cross-network update.
  - Better suited when you want minimal assumptions about the noise structure.

Evaluation:
  No trusted test set (noisy labels throughout).
  Proxy metric: net1–net2 prediction agreement on the training set.
    Rising   → networks converging on consistent predictions (good)
    Dropping → co-training destabilising (reduce noise_rate or increase T_k)
"""

import json
import os
import pickle

import pandas as pd
import torch
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

import lightning_wrapper
import utils
import train

torch.set_float32_matmul_precision("medium")


# =====================================================================
# Helpers
# =====================================================================

def _per_sample_ce(logits, labels):
    """Per-sample CE loss (binary [B,1] or multi-class [B,C])."""
    if logits.shape[-1] == 1:
        return F.binary_cross_entropy_with_logits(
            logits.squeeze(-1), labels.float(), reduction='none')
    return F.cross_entropy(logits, labels, reduction='none')


def _keep_ratio(epoch, warmup, noise_rate, T_k):
    """
    Fraction of small-loss samples kept per mini-batch.

    Timeline:
      [0, warmup)           → 1.0  (all samples, warmup period)
      [warmup, warmup+T_k)  → linearly decays from 1.0 to (1 - noise_rate)
      [warmup+T_k, ...]     → constant (1 - noise_rate)

    Args:
      noise_rate: assumed label noise rate (0–1).
                  ↑ larger → smaller keep set → more aggressive noise filtering
      T_k:        ramp length in epochs.
                  ↑ larger → slower, gentler ramp
    """
    if epoch < warmup:
        return 1.0
    t = epoch - warmup
    return max(1.0 - noise_rate * t / max(T_k, 1), 1.0 - noise_rate)


# =====================================================================
# Training / evaluation
# =====================================================================

def train_coteaching_epoch(net1, net2, opt1, opt2, loader,
                           keep_ratio, device, class0_mode='keep', log_fn=None):
    """
    One co-teaching epoch.

    Each mini-batch step:
      - net1 & net2 do independent forward passes on the same batch.
      - net1's small-loss indices  → net2 is updated on those samples.
      - net2's small-loss indices  → net1 is updated on those samples.

    class0_mode controls how class-0 (NTB / negative) samples are handled:
      'keep'   → all samples used in co-teaching (default)
      'remove' → class-0 samples dropped from every mini-batch before training;
                 only class-1 (TB) samples are used for the co-teaching update

    Returns (avg_loss_net1, avg_loss_net2).
    """
    net1.cuda()
    net2.cuda()
    net1.train()
    net2.train()
    total1 = total2 = 0.0
    n_batches = 0

    for batch_idx, batch in enumerate(loader):
        _, audio, _, masks, dse_ids, _ = batch
        audio  = audio.to(device)
        masks  = masks.to(device)
        labels = torch.argmax(dse_ids.float(), dim=1).to(device)

        if class0_mode == 'remove':
            keep = labels != 0
            if keep.sum() < 2:   # need ≥ 2 samples for a meaningful update
                continue
            audio  = audio[keep]
            masks  = masks[keep]
            labels = labels[keep]

        N      = labels.size(0)
        n_keep = max(1, int(keep_ratio * N))

        # Independent forward passes (separate computation graphs)
        logits1 = net1(audio, attention_mask=masks)['disease_logits']
        logits2 = net2(audio, attention_mask=masks)['disease_logits']

        loss1_per = _per_sample_ce(logits1, labels)
        loss2_per = _per_sample_ce(logits2, labels)

        # Each net identifies its own small-loss (clean) samples
        with torch.no_grad():
            _, clean1 = loss1_per.topk(n_keep, largest=False)
            _, clean2 = loss2_per.topk(n_keep, largest=False)

        # Cross-train:
        #   net2 learns from samples that net1 found easy (clean1)
        #   net1 learns from samples that net2 found easy (clean2)
        loss_net2 = loss2_per[clean1].mean()
        loss_net1 = loss1_per[clean2].mean()

        opt2.zero_grad()
        loss_net2.backward()
        opt2.step()

        opt1.zero_grad()
        loss_net1.backward()
        opt1.step()

        total1 += loss_net1.item()
        total2 += loss_net2.item()
        n_batches += 1

        if log_fn and batch_idx % 50 == 0:
            log_fn(f"    [{batch_idx:4d}/{len(loader)}]  "
                   f"loss1={loss_net1.item():.3f}  loss2={loss_net2.item():.3f}  "
                   f"keep={n_keep}/{N}  ratio={keep_ratio:.3f}")

    return total1 / max(n_batches, 1), total2 / max(n_batches, 1)


def eval_agreement(net1, net2, loader, device, log_fn=None):
    """
    Compute net1–net2 prediction agreement on the training set.

    Used as a proxy convergence metric when test labels are unreliable.
    Higher agreement → more stable co-training; watch for it plateauing or dropping.

    loader uses CoughDatasetsCollate format:
      (wav_names, wav_padded, None, attention_masks, dse_ids, [...])
    Returns agreement rate in [0, 1].
    """
    net1.eval()
    net2.eval()
    agree = total = 0

    with torch.no_grad():
        for _, audio, _, masks, dse_ids, _ in loader:
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
            total += dse_ids.size(0)

    rate = agree / max(total, 1)
    if log_fn:
        log_fn(f"  Net1–Net2 agreement: {agree}/{total} = {rate:.4f}")
    return rate


def save_label_analysis(net1, net2, df_train, hps, noisy_labels,
                        save_path, device, use_precomputed, precomputed_dir,
                        log_fn=None):
    """
    Run ensemble inference on the full training set and write per-sample analysis to CSV.

    Columns
    -------
    noisy_label   : original label from the dataset (may be wrong)
    pred_prob_tb  : ensemble P(TB) from the final net1+net2 models
    pred_label    : hard predicted label (1=TB, 0=NTB) at threshold 0.5

    How to read the output
    ----------------------
    - pred_label  : the denoised label suggested by the trained ensemble
    - pred_prob_tb close to 0 or 1 → model is confident; close to 0.5 → uncertain
    - pred_label ≠ noisy_label → model disagrees with original label (likely noisy)
    """
    # Import dataset utilities from train_dividemix to avoid duplicating eval logic
    from train_dividemix import DivideMixDataset, _dm_all_collate

    eval_ds = DivideMixDataset(
        df_train, hps, mode='all', train_mode=False,
        use_precomputed=use_precomputed, precomputed_dir=precomputed_dir,
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=hps.train.batch_size, shuffle=False,
        num_workers=4, collate_fn=_dm_all_collate, pin_memory=True,
    )

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

    pred_np = pred_probs.numpy()
    result = pd.DataFrame({
        'noisy_label':  noisy_labels,
        'pred_prob_tb': pred_np,
        'pred_label':   (pred_np > 0.5).astype(int),
    })
    result.to_csv(save_path, index=False)

    if log_fn:
        n_pred_tb  = int(result['pred_label'].sum())
        n_noisy_tb = int((noisy_labels == 1).sum())
        n_changed  = int((result['pred_label'].values != noisy_labels).sum())
        log_fn(f"  Label analysis saved → {save_path}")
        log_fn(f"  Pred TB (class 1) : {n_pred_tb}  (noisy TB: {n_noisy_tb})")
        log_fn(f"  Label changes     : {n_changed} samples differ from noisy_label")


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

    # --- Co-Teaching hyper-parameters ---
    ct_config = {
        # Warmup: standard CE training before co-teaching begins.
        # ↑ more  → cleaner network initialisation, longer total training
        # ↓ fewer → faster start, noisier initialisation
        "warmup": 10,

        # Total training epochs (warmup + co-teaching).
        # ↑ more  → longer convergence window
        # ↓ fewer → faster, may underfit or not converge
        "num_epochs": 300,

        # Assumed label noise rate (fraction of noisy labels in the dataset).
        # ↑ larger → smaller keep set per batch, more aggressive filtering
        # ↓ smaller → closer to standard training (keep almost all samples)
        # Rule of thumb: set to your estimated noise fraction (e.g. 0.2 for 20 % noise).
        "noise_rate": 0.7,

        # Ramp length: how many epochs (after warmup) to linearly increase R from 0 to noise_rate.
        # ↑ more  → slower, gentler increase of the noise filter
        # ↓ fewer → noise filter kicks in more aggressively early on
        "T_k": 10,

        # How class-0 (NTB / negative) samples are handled during co-teaching:
        # 'keep'   → all samples participate in training (default)
        # 'remove' → class-0 dropped from every mini-batch; only TB samples used
        #            (use when you want the model to focus purely on TB detection
        #             and do not trust negative labels at all)
        "class0_mode": "keep",
    }
    with open(os.path.join(model_dir, "ct_config.json"), "w") as f:
        json.dump(ct_config, f, indent=2)

    # --- Data ---
    df_train, _ = train.load_data(hps)
    collate_fn    = train.get_collate_fn(hps)
    target_labels = df_train[hps.data.target_column]

    if not args.use_precomputed:
        utils.compute_spectrogram_stats_from_dataset(
            df_train, hps.data,
            pickle_path=f"{hps.model_dir}/wav_stats.pickle",
        )

    logger_py = utils.get_logger(hps.model_dir)
    logger_py.info(hps)
    logger_py.info(f"Co-Teaching config: {ct_config}")

    # Validation holdout (warmup early-stopping only).
    # Note: no trusted test set — evaluation uses net1–net2 agreement as proxy metric.
    sgkf = StratifiedGroupKFold(n_splits=5, random_state=42, shuffle=True)
    wu_trn_idx, wu_val_idx = next(
        sgkf.split(df_train, target_labels, df_train["participant"])
    )
    df_wu_trn = df_train.iloc[wu_trn_idx].reset_index(drop=True)
    df_wu_val = df_train.iloc[wu_val_idx].reset_index(drop=True)

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
    # =========================================================
    logger_py.info(f"\n{'='*20} Warmup ({ct_config['warmup']} epochs) {'='*20}")

    wu_train_loader, wu_val_loader = train.prepare_fold_data(
        df_wu_trn, df_wu_val, hps, collate_fn,
        use_precomputed=args.use_precomputed,
        precomputed_dir=args.precomputed_dir,
    )
    tb_logger = TensorBoardLogger(hps.model_dir, name="coteaching")

    for net, tag in [(net1, "net1"), (net2, "net2")]:
        logger_py.info(f"  Warmup {tag}")
        runner    = lightning_wrapper.CoughClassificationRunner(net, hps, logger_py)
        ckpt_cb   = ModelCheckpoint(
            dirpath=f"{hps.model_dir}/warmup_{tag}",
            monitor="val/loss", filename="best", save_top_k=1, mode="min")
        early_stop = EarlyStopping(monitor="val/loss", patience=5, mode="min")
        trainer = L.Trainer(
            max_epochs=ct_config["warmup"],
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
    # Phase 2 – Co-teaching  (manual loop)
    # =========================================================
    logger_py.info(f"\n{'='*20} Co-teaching {'='*20}")

    # Use full training set for co-teaching (not just the warmup split)
    train_loader, _ = train.prepare_fold_data(
        df_wu_trn, df_wu_val, hps, collate_fn,
        use_precomputed=args.use_precomputed,
        precomputed_dir=args.precomputed_dir,
    )

    metrics_history = {}

    for epoch in range(ct_config["warmup"], ct_config["num_epochs"] + 1):
        kr = _keep_ratio(epoch, ct_config["warmup"],
                         ct_config["noise_rate"], ct_config["T_k"])
        logger_py.info(
            f"\n{'='*20} Epoch {epoch}/{ct_config['num_epochs']}  keep_ratio={kr:.3f} {'='*20}")

        l1, l2 = train_coteaching_epoch(
            net1, net2, opt1, opt2, train_loader, kr, device,
            class0_mode=ct_config["class0_mode"], log_fn=logger_py.info)
        logger_py.info(f"  avg_loss: net1={l1:.4f}  net2={l2:.4f}")

        # --- Proxy metric: net1–net2 prediction agreement ---
        # Rising   → networks converging on consistent predictions (good)
        # Dropping → co-training destabilising (check noise_rate / T_k)
        agreement = eval_agreement(net1, net2, train_loader, device, log_fn=logger_py.info)

        epoch_metrics = {
            "epoch":        epoch,
            "keep_ratio":   kr,
            "avg_loss_net1": l1,
            "avg_loss_net2": l2,
            "agreement":    agreement,
        }
        metrics_history[epoch] = epoch_metrics

        if epoch % 10 == 0:
            torch.save(net1.state_dict(), f"{hps.model_dir}/net1_ep{epoch:04d}.pt")
            torch.save(net2.state_dict(), f"{hps.model_dir}/net2_ep{epoch:04d}.pt")

    torch.save(net1.state_dict(), f"{hps.model_dir}/net1_final.pt")
    torch.save(net2.state_dict(), f"{hps.model_dir}/net2_final.pt")
    with open(os.path.join(hps.model_dir, "ct_metrics.pkl"), "wb") as f:
        pickle.dump(metrics_history, f)

    # --- Save label analysis (denoised labels) ---
    logger_py.info("\nGenerating label analysis...")
    save_label_analysis(
        net1, net2, df_train, hps,
        noisy_labels=df_train[hps.data.target_column].values,
        save_path=os.path.join(hps.model_dir, "label_analysis.csv"),
        device=device,
        use_precomputed=args.use_precomputed,
        precomputed_dir=args.precomputed_dir,
        log_fn=logger_py.info,
    )

    logger_py.info("Co-Teaching training complete.")


if __name__ == "__main__":
    main()
