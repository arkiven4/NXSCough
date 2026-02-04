import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning as L
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from peft import PeftModel
from sklearn.metrics import (
    confusion_matrix, accuracy_score,
    balanced_accuracy_score,
    roc_auc_score, roc_curve
)

import commons
import models
import utils
import loss_functions


class CoughClassificationRunner(L.LightningModule):
    def __init__(self, model, hps, custom_logger, class_weights=[], probs_threshold=0.5):
        super().__init__()
        self.model = model
        self.hps = hps
        self.custom_logger = custom_logger
        self.class_weights = class_weights
        self.probs_threshold = probs_threshold
        self.calibrate_threshold = False
        self.generate_figure = False
        self.loss_fn = loss_functions.get_losses_fn(hps.train.loss_function)

        # =============================================================
        # SECTION: Additional Setupo
        # =============================================================
        ssl_model = None
        if hps.model.pooling_model.split("_")[0] == "WavLMEncoder":
            self.custom_logger.info("Loaded Pretrained WavLM")
            from transformers import AutoModel
            ssl_model = AutoModel.from_pretrained("microsoft/wavlm-large")
            self.model.feature_extractor.load_state_dict(
                ssl_model.feature_extractor.state_dict())
            self.model.feature_extractor._freeze_parameters()
            del ssl_model
            torch.cuda.empty_cache()

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_percentage = 100 * trainable_params / total_params if total_params > 0 else 0
        self.custom_logger.info(
            f'Trainable params: {trainable_params} | Total params: {total_params} | Trainable%: {trainable_percentage:.2f}% | Size: {trainable_params/(1e6):.2f}M')

    def forward(self, x1, x2=None, attention_mask=None, tabular_ids=None, train=False):
        x = self.model(x1, x2=x2, attention_mask=attention_mask, tabular_ids=tabular_ids, train=train)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW([
            {"params": self.model.parameters(), "lr": self.hps.train.learning_rate,
             "weight_decay": self.hps.train.weight_decay},
            # {"params": self.model.sscl_model.parameters(), "lr": 1e-5},
            # {"params": self.model.head.parameters(), "lr": self.hps.train.learning_rate},
            # {"params": self.center_loss.parameters(), "lr": self.hps.train.learning_rate},
        ])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=2)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
            },
        }

    # After training, the model have PEFT, but the ckpt dont have PEFT, so it will brreak
    # def on_save_checkpoint(self, checkpoint):
    #     import copy
    #     temp_model = copy.deepcopy(self.model)
    #     backbone = self.model.backbone_model
    #     if not isinstance(backbone, PeftModel):
    #         return
    #     merged = self.model.backbone_model.merge_and_unload()
    #     temp_model.backbone_model = merged
    #     checkpoint["state_dict"] = temp_model.state_dict()

    #     state_dict = checkpoint["state_dict"]
    #     fixed_state_dict = {}
    #     for k, v in state_dict.items():
    #         k = k.replace(".base_model.model", "")
    #         if not k.startswith("model."):
    #             fixed_state_dict[f"model.{k}"] = v
    #         else:
    #             fixed_state_dict[k] = v
    #     checkpoint["state_dict"] = fixed_state_dict

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
        _, audio1, audio2, attention_masks, dse_ids, [patient_ids, _, tabular_ids, _] = batch
        out_model = self.forward(audio1, audio2, attention_mask=attention_masks, tabular_ids=tabular_ids, train=True)

        tot_loss = []
        if "disease_logits" in out_model:
            ld = self.loss_fn(out_model, batch)
            tot_loss.append(ld[0])

        if hasattr(self.model, "calc_additional_loss"):
            tot_loss.append(self.model.calc_additional_loss(out_model, batch))

        if "loss" in out_model:
            tot_loss.append(out_model["loss"])

        # if "embedding" in out_model:
        #     tot_loss.append(self.supcon_loss(out_model["embedding"], dse_ids) * 0.03)
        #     tot_loss.append(self.center_loss(out_model["embedding"], dse_ids) * 0.005)

        loss = sum(tot_loss)
        for idx_loss, now_loss in enumerate(tot_loss):
            self.log(f"train/loss_{idx_loss}", now_loss, on_step=True,
                     on_epoch=False, prog_bar=False, logger=True)

        self.log("train/loss_step", loss, on_step=True,
                 on_epoch=False, prog_bar=True, logger=True)
        self.logger.experiment.add_scalars(
            'loss', {'train': loss}, self.global_step)

        if "pred" in out_model:
            random_idx = torch.randint(
                0, out_model["pred"].size(0), (1,)).item()

            self.logger.experiment.add_image(
                "pred_image",
                utils.plot_spectrogram_to_numpy(
                    out_model["pred"][random_idx].data.cpu().numpy().T),
                global_step=self.global_step
            )

            self.logger.experiment.add_image(
                "orig_image",
                utils.plot_spectrogram_to_numpy(
                    audio1[random_idx].data.cpu().numpy()),
                global_step=self.global_step
            )

        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        lr = self.optimizers().param_groups[0]["lr"]
        self.log("train/lr", lr, sync_dist=True)

    # def on_train_epoch_end(self):
    #     self.loss_fn.step_lambda(factor=1.1)
    #     self.log("lambda", self.loss_fn.lambda_val, prog_bar=True, sync_dist=True,)
        
    def validation_step(self, batch, batch_idx):
        _, audio1, audio2, attention_masks, dse_ids, [patient_ids, _, tabular_ids, _] = batch
        out_model = self.forward(audio1, audio2, attention_mask=attention_masks, tabular_ids=tabular_ids)

        tot_loss = []
        if "disease_logits" in out_model:
            ld = self.loss_fn(out_model, batch)
            tot_loss.append(ld[0])

        if hasattr(self.model, "calc_additional_loss"):
            tot_loss.append(self.model.calc_additional_loss(out_model, batch))

        if "loss" in out_model:
            tot_loss.append(out_model["loss"])

        # if "embedding" in out_model:
        #     tot_loss.append(self.supcon_loss(out_model["embedding"], dse_ids) * 0.03)
        #     tot_loss.append(self.center_loss(out_model["embedding"], dse_ids) * 0.005)

        loss = sum(tot_loss)
        self.logger.experiment.add_scalars('loss', {'valid': loss}, self.global_step)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)

    def on_test_epoch_start(self):
        self.test_preds = []
        self.test_labels = []
        self.test_probs = []

    def test_step(self, batch, batch_idx):
        _, audio1, audio2, attention_masks, dse_ids, [_, _, tabular_ids, _] = batch
        logits = self.forward(audio1, audio2, attention_mask=attention_masks, tabular_ids=tabular_ids)["disease_logits"]

        dse_ids = dse_ids.float()
        labels = torch.argmax(dse_ids, dim=1)
        if logits.shape[-1] == 1:
            probs = torch.sigmoid(logits)     # [B]
            preds = (probs >= self.probs_threshold).long()     # [B]
        else:
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

        self.test_preds.append(preds.cpu())
        self.test_labels.append(labels.cpu())
        self.test_probs.append(probs.cpu())

    def on_test_epoch_end(self):
        preds = torch.cat(self.test_preds).numpy()
        labels = torch.cat(self.test_labels).numpy()
        probs = torch.cat(self.test_probs).numpy()

        if self.calibrate_threshold:
            optimized_threshold = utils.optimize_threshold_youden(labels, probs)
            self.probs_threshold = optimized_threshold
            preds = (probs >= self.probs_threshold).astype(int)

        # -----------------------------------------
        # Confusion Matrix + metrics
        # -----------------------------------------
        cm = confusion_matrix(labels, preds, labels=[0, 1])
        n_classes = cm.shape[0]

        # sanity check: binary only
        assert cm.shape == (2, 2), f"Expected binary confusion matrix, got {cm.shape}"

        TN, FP = cm[0, 0], cm[0, 1]
        FN, TP = cm[1, 0], cm[1, 1]

        acc = accuracy_score(labels, preds)

        # clinical metrics
        sens = TP / (TP + FN) if (TP + FN) > 0 else 0.0   # TB recall
        spec = TN / (TN + FP) if (TN + FP) > 0 else 0.0   # non-TB recall

        # optional but valid
        b_acc = 0.5 * (sens + spec)

        # -----------------------------------------
        # AUROC + KPI pAUROC (≥80% sens, ≥60% spec)
        # -----------------------------------------
        # unique_labels = np.unique(labels)
        # if unique_labels.size == 2:
        if n_classes == 2:
            auroc = roc_auc_score(labels, probs)

            fpr, tpr, thresholds = roc_curve(labels, probs)
            spec_curve = 1 - fpr

            mask = (tpr >= 0.80) & (spec_curve >= 0.60)

            if mask.sum() > 1:
                p_auroc = np.trapz(tpr[mask], fpr[mask])
            else:
                p_auroc = 0.0
        else:
            auroc = 0.0
            p_auroc = 0.0

        # -----------------------------------------
        # Log
        # -----------------------------------------
        self.log("test_acc", acc, sync_dist=True)
        self.log("test_bacc", b_acc, sync_dist=True)
        self.log("test_sens", sens, sync_dist=True)
        self.log("test_spec", spec, sync_dist=True)
        self.log("test_auroc", auroc, sync_dist=True)
        self.log("test_pauroc", p_auroc, sync_dist=True)

        # -----------------------------------------
        # Save CM
        # -----------------------------------------
        if self.generate_figure:
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Non_TB", "TB"], yticklabels=["Non_TB", "TB"])
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title("Confusion Matrix")
            plt.savefig(f"{self.hps.model_dir}/result_cm.png")
            plt.close()

        # -----------------------------------------
        # ROC Quadrant Analysis + KPI Fail Overlay
        # -----------------------------------------
        if n_classes == 2 and self.generate_figure:
            target_sens = 0.80
            target_spec = 0.60
            max_fpr_allowed = 1 - target_spec

            # ROC curve
            fpr, tpr, thresholds = roc_curve(labels, probs)

            # KPI mask
            fail_mask = (tpr < target_sens) | (fpr > max_fpr_allowed)

            # ---- FIGURE 1: ROC Quadrant Analysis ----
            plt.figure(figsize=(7, 7))
            plt.plot(fpr, tpr, label=f"ROC (AUC={auroc:.3f})", linewidth=2)

            plt.axhline(target_sens, color="green",
                        linestyle="--", linewidth=1)
            plt.axvline(max_fpr_allowed, color="green",
                        linestyle="--", linewidth=1)

            plt.fill_betweenx(
                y=[target_sens, 1],
                x1=0,
                x2=max_fpr_allowed,
                alpha=0.15,
                color="green",
                label="KPI zone"
            )

            plt.xlabel("False Positive Rate (1 - Specificity)")
            plt.ylabel("True Positive Rate (Sensitivity)")
            plt.title("ROC Quadrant Analysis")
            plt.grid(alpha=0.4)
            plt.legend()

            plt.savefig(f"{self.hps.model_dir}/roc_quadrant.png",
                        dpi=200, bbox_inches="tight")
            plt.close()

        return {
            "acc": acc,
            "bacc": b_acc,
            "sens": sens,
            "spec": spec,
            "auroc": auroc,
            "pauroc": p_auroc,
        }


class CoughDetectionRunner(L.LightningModule):
    def __init__(self, model, hps, custom_logger, class_weights=[]):
        super().__init__()
        self.model = model
        self.hps = hps
        self.custom_logger = custom_logger
        self.class_weights = class_weights
        self.prev_cough_emb = None

        ssl_model = None
        if hps.model.pooling_model.split("_")[0] == "WavLMEncoder":
            self.custom_logger.info("Loaded Pretrained WavLM")
            from transformers import AutoModel
            ssl_model = AutoModel.from_pretrained("microsoft/wavlm-large")
            self.model.feature_extractor.load_state_dict(
                ssl_model.feature_extractor.state_dict())
            self.model.feature_extractor._freeze_parameters()
            del ssl_model
            torch.cuda.empty_cache()

        if hps.model.ssl_model_type.lower() == "wavlm":
            from wrapper.wavlm_plus import WavLMWrapper
            ssl_model = WavLMWrapper(hps.model)
            if ssl_model != None:
                trainable_params = sum(
                    p.numel() for p in ssl_model.parameters() if p.requires_grad)
                total_params = sum(p.numel() for p in ssl_model.parameters())
                trainable_percentage = 100 * trainable_params / \
                    total_params if total_params > 0 else 0
                self.custom_logger.info(
                    f'Trainable params: {trainable_params} | Total params: {total_params} | Trainable%: {trainable_percentage:.2f}% | Size: {trainable_params/(1e6):.2f}M')
                hps.model.feature_dim = ssl_model.hidden_size_ssl

            ssl_model.model_pooling = self.model
            self.model = ssl_model
            self.model.backbone_model.feature_extractor._freeze_parameters()

    def forward(self, x, attention_mask=None):
        x = self.model(x, attention_mask=attention_mask)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hps.train.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=2)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
            },
        }

    def on_after_backward(self):
        norm_type = 2
        total_norm = 0
        parameters = list(
            filter(lambda p: p.grad is not None, self.model.parameters()))
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)
        self.log("train/grad_norm", total_norm, sync_dist=True)

    def training_step(self, batch, batch_idx):
        _, audio, _, attention_masks, dse_ids, _ = batch

        out_model = self.forward(audio, attention_mask=attention_masks)
        tot_loss = []
        if "disease_logits" in out_model:
            ld = utils.many_loss_category(
                out_model["disease_logits"], dse_ids, loss_type=self.hps.train.loss_function, weights=self.class_weights)
            tot_loss.append(ld[0])

        if "loss" in out_model:
            tot_loss.append(out_model["loss"])

        # if "embedding" in out_model:
        #     tot_loss.append(self.supcon_loss(out_model["embedding"], dse_ids) * 0.03)
        #     tot_loss.append(self.center_loss(out_model["embedding"], dse_ids) * 0.005)

        loss = sum(tot_loss)
        for idx_loss, now_loss in enumerate(tot_loss):
            self.log(f"train/loss_{idx_loss}", now_loss, on_step=True,
                     on_epoch=False, prog_bar=False, logger=True)

        self.log("train/loss_step", loss, on_step=True,
                 on_epoch=False, prog_bar=True, logger=True)
        self.logger.experiment.add_scalars(
            'loss', {'train': loss}, self.global_step)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        lr = self.optimizers().param_groups[0]["lr"]
        self.log("train/lr", lr, sync_dist=True)

    def on_validation_epoch_start(self):
        self.validation_step_outputs = []

    def validation_step(self, batch, batch_idx):
        _, audio, _, attention_masks, dse_ids, othr_ids = batch
        out_model = self.forward(audio, attention_mask=attention_masks)
        tot_loss = []
        if "disease_logits" in out_model:
            ld = utils.many_loss_category(
                out_model["disease_logits"], dse_ids, loss_type=self.hps.train.loss_function, weights=self.class_weights)
            tot_loss.append(ld[0])
        if "loss" in out_model:
            tot_loss.append(out_model["loss"])
        loss = sum(tot_loss)

        self.logger.experiment.add_scalars(
            'loss', {'valid': loss}, self.global_step)
        self.log("val/loss", loss, on_step=False,
                 on_epoch=True, prog_bar=False, logger=True)

        # ---- store embeddings for Phase-1 geometry ----
        emb = out_model["embedding"].detach()
        labels = dse_ids.squeeze(-1).detach()
        self.validation_step_outputs.append({
            "emb": emb,
            "labels": labels
        })

    def on_validation_epoch_end(self):
        # ---- collect embeddings ----
        emb = torch.cat([x["emb"]
                        for x in self.validation_step_outputs], dim=0)
        labels = torch.cat([x["labels"]
                           for x in self.validation_step_outputs], dim=0)

        cough_emb = emb[labels == 1]
        bg_emb = emb[labels == 0]

        # Guardrail: skip if batch unlucky
        if cough_emb.size(0) < 5 or bg_emb.size(0) < 5:
            return

        compactness = torch.mean(torch.cdist(cough_emb, cough_emb)).item()
        intra = torch.mean(torch.cdist(cough_emb, cough_emb))
        inter = torch.mean(torch.cdist(cough_emb, bg_emb))
        margin = (inter / intra).item()
        drift = 0.0
        if self.prev_cough_emb is not None:
            drift = torch.norm(self.prev_cough_emb.mean(
                0) - cough_emb.mean(0)).item()
        self.prev_cough_emb = cough_emb.detach().clone()

        # ---- log geometry ----
        self.log("val/compactness", compactness, prog_bar=False)
        self.log("val/margin", margin, prog_bar=False)
        self.log("val/drift", drift, prog_bar=False)
        self.log("val/total_geometri", (compactness * 0.25) +
                 (2 - margin) + drift, prog_bar=False)

        # # TensorBoard (optional)
        # self.logger.experiment.add_scalars(
        #     "phase1_geometry",
        #     metrics,
        #     self.current_epoch
        # )

        # # ---- auto-stop signal ----
        # if self.phase1_monitor.should_stop():
        #     self.prev_stop_signal = True

    def on_test_epoch_start(self):
        self.test_preds = []
        self.test_labels = []

    def test_step(self, batch, batch_idx):
        _, audio, _, attention_masks, dse_ids, _ = batch

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
        # plt.figure(figsize=(6, 5))
        # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
        #             xticklabels=class_labels, yticklabels=class_labels)
        # plt.xlabel("Predicted")
        # plt.ylabel("Actual")
        # plt.title("Confusion Matrix")
        # plt.savefig(f"{self.hparams.model_dir}/result_cm.png")
        # plt.close()

        return {
            "acc": acc,
            "bacc": b_acc,
            "sens": sens,
            "spec": spec,
        }
