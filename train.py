import argparse, inspect, json, os, pickle, socket, subprocess, warnings, random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, auc, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score, roc_curve
from sklearn.utils import resample
from tensorboard.backend.event_processing import event_accumulator
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoConfig, AutoFeatureExtractor, AutoModel, get_linear_schedule_with_warmup

import commons, models, utils
from cough_datasets import CoughDatasets, CoughDatasetsCollate
from wrapper.wav2vec import Wav2VecWrapper
from wrapper.wavlm_plus import WavLMWrapper
from wrapper.whisper import WhisperWrapper

import warnings
warnings.simplefilter("ignore", UserWarning)

# =============================================================
# SECTION: Intialize Data
# =============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--init", action="store_true")
parser.add_argument("--model_name", type=str, default="try_wavlmlora_downstream")
parser.add_argument("--config_path", type=str, default="configs/ssl_finetuning.json")
parser.add_argument('--fraction', type=float, default=1.0, help='Fraction of data to sample (0-1, default=1.0)')
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

BATCH_SIZE = hps.train.batch_size
ACCUMULATION_STEP = hps.train.accumulation_steps
assert (ACCUMULATION_STEP > 0) and (BATCH_SIZE % ACCUMULATION_STEP == 0)
cur_bs = BATCH_SIZE // ACCUMULATION_STEP

# =============================================================
# SECTION: Loading Data
# =============================================================

df_longi = pd.read_csv('/run/media/fourier/Data1/Pras/DatabaseLLM/coda/longitudinal_original.csv')
df_solic = pd.read_csv('/run/media/fourier/Data1/Pras/DatabaseLLM/coda/solicited_original.csv')

participant_mapping_longi = {participant: idx for idx, participant in enumerate(set(np.concatenate([df_solic['participant'].unique(), df_longi['participant'].unique()])))}
df_longi['participant'] = df_longi['participant'].map(participant_mapping_longi)
df_solic['participant'] = df_solic['participant'].map(participant_mapping_longi)

gender_mapping_longi = {gender: idx for idx, gender in enumerate(df_longi['sex'].unique())}
df_longi['sex'] = df_longi['sex'].map(gender_mapping_longi)
df_solic['sex'] = df_solic['sex'].map(gender_mapping_longi)

df_longi_train, df_longi_val = utils.stratified_group_split(df_longi)
df_solic_train, df_solic_val = utils.stratified_group_split(df_solic)

df_train = pd.concat([df_longi_train, df_solic_train], ignore_index=True)
df_test = pd.concat([df_longi_val, df_solic_val], ignore_index=True)

if hps.data.reorder_target:
    cols = hps.data.column_order
    df_train = df_train[cols]
    df_test = df_test[cols]

disease_codes = df_train[hps.data.target_column].unique().tolist()
class_frequencies = df_train[hps.data.target_column].value_counts().to_dict()
total_samples = len(df_train)
class_weights = {cls: total_samples / (len(disease_codes) * freq) if freq != 0 else 0 for cls, freq in class_frequencies.items()}
weights_list = [class_weights[cls] for cls in disease_codes]
class_weights_tensor = torch.tensor(weights_list, device='cuda', dtype=torch.float)
print(class_weights_tensor)

# =============================================================
# SECTION: Setup Logger, Dataloader
# =============================================================
logger = utils.get_logger(hps.model_dir)
logger.info(hps)

writer = SummaryWriter(log_dir=hps.model_dir)
writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

collate_fn = CoughDatasetsCollate(hps.data.many_class)
train_dataset = CoughDatasets(df_train.values, hps.data, train=True)
val_dataset = CoughDatasets(df_test.values, hps.data, train=False)

train_loader = DataLoader(train_dataset, num_workers=28, shuffle=True, batch_size=cur_bs, pin_memory=True, drop_last=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, num_workers=28, shuffle=False, batch_size=hps.train.batch_size, pin_memory=True, drop_last=True, collate_fn=collate_fn)

print(next(iter(train_loader))[1][0].numpy().shape)
# =============================================================
# SECTION: Setup Logger, Dataloader
# =============================================================
logger.info(f"======================================")
logger.info(f"✨ Loss: {hps.train.loss_function}")
logger.info(f"✨ Use Between Class Training: {hps.data.mix_audio}")
logger.info(f"✨ Use Augment: {hps.data.augment_data}")
logger.info(f"✨ Use Rawboost Augment: {hps.data.augment_rawboost}")
logger.info(f"✨ Padding Type: {hps.data.pad_types}")
logger.info(f"✨ Using Model: {hps.model.pooling_model}")
logger.info(f"✨ Tensorboard: http://100.101.198.75:{port}")
logger.info(f"======================================")

logger.info(f"Total Training Data : {len(df_train)}")

epoch_str = 1
global_step = 0
num_training_steps = len(train_loader) * 20
num_warmup_steps = int(0.01 * num_training_steps)  # 5% warmup

ssl_model_type = hps.model.ssl_model_type.lower()
ssl_model = None
if ssl_model_type == "wav2vec2":
    ssl_model = Wav2VecWrapper(hps.model)
    logger.info("✨ Using Wav2Vec2 SSL Model")
elif ssl_model_type == "wavlm":
    ssl_model = WavLMWrapper(hps.model)
    logger.info("✨ Using WavLM SSL Model")
elif ssl_model_type == "whisper":
    ssl_model = WhisperWrapper(hps.model)
    logger.info("✨ Using Whisper SSL Model")
elif ssl_model_type == "nonssl":
    logger.info("✨ Using Non-SSL Model")

if ssl_model != None:
    trainable_params = sum(p.numel() for p in ssl_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in ssl_model.parameters())
    trainable_percentage = 100 * trainable_params / total_params if total_params > 0 else 0
    logger.info(f'Trainable params: {trainable_params} | Total params: {total_params} | Trainable%: {trainable_percentage:.2f}% | Size: {trainable_params/(1e6):.2f}M')
    hps.model.feature_dim = ssl_model.hidden_size_ssl
hps.model.spk_dim = len(participant_mapping_longi)

pool_net = getattr(models, hps.model.pooling_model)
pool_model = pool_net(hps.model.feature_dim, **hps.model)

if ssl_model != None:
    ssl_model.model_pooling = pool_model
    pool_model = ssl_model

pool_model = pool_model.cuda()
optimizer_p = torch.optim.AdamW(list(filter(lambda p: p.requires_grad, pool_model.parameters())), hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)
scheduler_p = get_linear_schedule_with_warmup(
    optimizer_p, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

class_code_pool_net = inspect.getsource(pool_net)
with open(f'{hps.model_dir}/model_net.py.bak', 'w') as f:
    f.write("import torch\nimport torch.nn as nn\n\n")
    f.write(class_code_pool_net)

# =============================================================
# SECTION: Additional Setup
# =============================================================
if hps.model.pooling_model.split("_")[0] == "WavLMEncoder":
    print("Loaded Pretrained WavLM")
    ssl_model = AutoModel.from_pretrained("microsoft/wavlm-large")
    pool_model.feature_extractor.load_state_dict(ssl_model.feature_extractor.state_dict())
    pool_model.feature_extractor._freeze_parameters()
    del ssl_model

# # =============================================================
# # SECTION: Setup Logger, Dataloader
# # =============================================================
best_lost = np.inf
patience_val = []

try:
    _, _, _, _, epoch_str = utils.load_checkpoint(
        utils.latest_checkpoint_path(hps.model_dir, "pool_*.pth"),
        pool_model,
        optimizer_p,
        scheduler_p,
    )

    epoch_str += 1
    global_step = (epoch_str - 1) * len(train_loader)

    with open(os.path.join(hps.model_dir, "traindata.pickle"), 'rb') as handle:
        traindata = pickle.load(handle)
        best_lost = traindata['best_lost']
        patience_val = traindata['patience_val']
    
except Exception as e:
    print(e)

scaler = GradScaler('cuda')
optimizer_p.zero_grad(set_to_none=True)

# =============================================================
# SECTION: Train Epoch
# =============================================================
for epoch in range(epoch_str, hps.train.epochs + 1):
    pool_model.train()

    batch_cnt = 0
    for batch_idx, (wav_names, audio, attention_masks, dse_ids, othr_ids) in enumerate(tqdm(train_loader)):
        audio = audio.cuda(non_blocking=True).float().squeeze(1)
        attention_masks = attention_masks.cuda(non_blocking=True).float()
        dse_ids = dse_ids.cuda(non_blocking=True).float()
        spk_ids = othr_ids[0].cuda(non_blocking=True).long()
        gndr_ids = othr_ids[1].cuda(non_blocking=True).long()

        x_lengths = torch.tensor(commons.compute_length_from_mask(attention_masks)).cuda(non_blocking=True)
        with torch.amp.autocast("cuda", enabled=True):
            out_model = pool_model(audio, attention_mask=attention_masks, grl_lambda=hps.model.grl_lambda)
        
            ld = utils.many_loss_category(out_model["disease_logits"], dse_ids, loss_type=hps.train.loss_function, weights=class_weights_tensor, model=pool_model)
            logit_keys = [k for k in out_model.keys() if k.endswith("_logits") and k != "disease_logits"]
            loss_cfg = [(0.5, spk_ids), (0.5, gndr_ids)]
            aux_losses = {}
            for (key, (w, target)) in zip(logit_keys, loss_cfg):
                aux_losses[key] = utils.many_loss_category(
                    out_model[key],
                    target,
                    loss_type=hps.train.loss_function,
                    model=pool_model
                )[0] * w
            l_ortho = utils.orthogonality_loss(out_model["d_emb"], out_model["s_emb"]) if {"d_emb", "s_emb"} <= out_model.keys() else 0
            l_internal = out_model.get("internal_loss", 0)

            if 'x_rec' in out_model:
                recon_loss = F.mse_loss(out_model['x_rec'], audio, reduction='mean')
                kl = (-0.5 * torch.sum(1 + out_model['logvar']  - out_model['mu'].pow(2) - out_model['logvar'].exp(), dim=1)).mean() * 1.0
                
            # Backprog
            total_aux_loss = sum(aux_losses.values()) if aux_losses else 0
            loss_g = sum(ld + [total_aux_loss] + [l_ortho] + [l_internal]) / ACCUMULATION_STEP #  + 

        # Logging
        aux_losses = [v.item() for v in aux_losses.values()]
        loss_gs = ld + aux_losses + [l_ortho] + [l_internal] # ld + aux_losses + [l_ortho] + [l_internal] + 

        scaler.scale(loss_g).backward()
        grad_norm = commons.clip_grad_value_(pool_model.parameters(), None)
        
        if (batch_cnt + 1) % ACCUMULATION_STEP == 0 or (batch_cnt + 1) == len(train_loader):
            scaler.step(optimizer_p)
            scaler.update() 
            scheduler_p.step()
            optimizer_p.zero_grad(set_to_none=True)
        
        batch_cnt = batch_cnt + 1
        if batch_idx % hps.train.log_interval == 0:
            scalar_dict = {
                "loss/g/total": loss_g,
                "info/learning_rate": scheduler_p.get_last_lr()[0],
                "info/grad_norm": grad_norm,
            }

            scalar_dict.update(
                {"loss/g/{}".format(i): v for i, v in enumerate(loss_gs)}
            )

            utils.summarize(
                writer=writer,
                global_step=global_step,
                scalars=scalar_dict,
            )

            if "x_rec" in out_model:
                idx = random.randint(0, audio.size(0) - 1)

                orig = audio[idx:idx+1]      # [1, 1, T, M]
                recon = out_model["x_rec"][idx:idx+1]  # [1, 1, T, M]

                # Normalize 0–1
                orig = (orig - orig.min()) / (orig.max() - orig.min() + 1e-8)
                recon = (recon - recon.min()) / (recon.max() - recon.min() + 1e-8)

                writer.add_image("VAE/original", orig, global_step)
                writer.add_image("VAE/recon", recon, global_step)

        global_step += 1
        
    pool_model.eval()
    losses_tot = []
    acc_tot = []
    with torch.no_grad():
        for batch_idx, (wav_names, audio, attention_masks, dse_ids, spk_ids) in enumerate(tqdm(val_loader)):
            audio = audio.cuda(non_blocking=True).float().squeeze(1)
            attention_masks = attention_masks.cuda(non_blocking=True).float()
            dse_ids = dse_ids.cuda(non_blocking=True).float()
            spk_ids = othr_ids[0].cuda(non_blocking=True).long()
            gndr_ids = othr_ids[1].cuda(non_blocking=True).long()

            x_lengths = torch.tensor(commons.compute_length_from_mask(attention_masks)).cuda(non_blocking=True).long()
            out_model = pool_model(audio, attention_mask=attention_masks)

            if 'x_rec' in out_model:
                recon_loss = F.mse_loss(out_model['x_rec'], audio, reduction='mean')
                kl = (-0.5 * torch.sum(1 + out_model['logvar']  - out_model['mu'].pow(2) - out_model['logvar'].exp(), dim=1)).mean() * 1.0
            loss_gs = utils.many_loss_category(out_model["disease_logits"], dse_ids, loss_type=hps.train.loss_function, weights=class_weights_tensor, model=pool_model)

            loss_g = sum(loss_gs)
            if batch_idx == 0:
                losses_tot = loss_gs
            else:
                losses_tot = [x + y for (x, y) in zip(losses_tot, loss_gs)]

    losses_tot = [x / len(val_loader) for x in losses_tot]
    loss_tot = torch.mean(torch.tensor(losses_tot)) #sum(losses_tot)
    scalar_dict = {"loss/g/total": loss_tot}
    scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_tot)})
    utils.summarize(
        writer=writer_eval, global_step=global_step, scalars=scalar_dict
    )

    pool_model.train()
    #scheduler_p.step()

    if loss_tot < best_lost and loss_tot > 0:
        logger.info(f"Get Best New Validation!!!!")
        best_lost = loss_tot
        patience_val = []
        utils.save_checkpoint(
            pool_model, optimizer_p, scheduler_p,
            hps.train.learning_rate, epoch,
            os.path.join(hps.model_dir, "best_pool.pth"))
    else:
        patience_val.append(1)
        logger.info(f"Patience: {len(patience_val)}")
        if len(patience_val) > 4:
            break


    logger.info("====> Epoch: {}".format(epoch))
    if epoch % 1 == 0:
        utils.save_checkpoint(
            pool_model, optimizer_p, scheduler_p,
            hps.train.learning_rate, epoch,
            os.path.join(hps.model_dir, "pool_{}.pth".format(epoch)))

        with open(os.path.join(hps.model_dir, "traindata.pickle"), 'wb') as handle:
            pickle.dump({
                "best_lost": best_lost,
                "patience_val": patience_val
            }, handle, protocol=pickle.HIGHEST_PROTOCOL)


# ######################################################################
# #
# # Metric Best Model
# #
# ######################################################################
val_dataset = CoughDatasets(df_test.values, hps.data, train=False)
val_loader = DataLoader(val_dataset, num_workers=28, shuffle=False, batch_size=hps.train.batch_size, pin_memory=True, drop_last=True, collate_fn=collate_fn)


_, _, _, _, epoch_str = utils.load_checkpoint(
    os.path.join(hps.model_dir, "best_pool.pth"),
    pool_model,
    None,
    None,
)

def evaluate_model(loader, split_name):
    pool_model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for _, (_, audio, attention_masks, dse_ids, _) in enumerate(tqdm(loader, desc=f"Eval {split_name}")):
            audio = audio.cuda(non_blocking=True).float().squeeze(1)
            attention_masks = attention_masks.cuda(non_blocking=True).float()
            dse_ids = dse_ids.cuda(non_blocking=True).float()

            logits = pool_model(audio, attention_mask=attention_masks)["disease_logits"]
            preds = torch.argmax(logits, dim=1)
            labels = np.argmax(dse_ids.cpu().numpy(), axis=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels)

    all_labels, all_preds = np.array(all_labels), np.array(all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    n_classes = cm.shape[0]
    class_labels = [f"Class {i+1}" for i in range(n_classes)]

    acc = accuracy_score(all_labels, all_preds)
    b_acc = balanced_accuracy_score(all_labels, all_preds)
    sens = np.mean([cm[i, i] / cm[i, :].sum() for i in range(n_classes) if cm[i, :].sum() > 0])
    spec = np.mean([(cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]) / (cm.sum() - cm[i, :].sum())
                    for i in range(n_classes) if (cm.sum() - cm[i, :].sum()) > 0])

    if split_name == "val":
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_labels, yticklabels=class_labels)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.savefig(f"{model_dir}/result_cm.png")

    return acc, b_acc, sens, spec

train_metrics = evaluate_model(train_loader, "train")
val_metrics = evaluate_model(val_loader, "val")

with open(f"{model_dir}/result_summary.txt", "w") as f:
    f.write(
        f"Train - Acc {train_metrics[0]:.2f} | BalAcc {train_metrics[1]:.2f} | "
        f"Sens {train_metrics[2]:.2f} | Spec {train_metrics[3]:.2f}\n"
    )
    f.write(
        f"Val - Acc {val_metrics[0]:.2f} | BalAcc {val_metrics[1]:.2f} | "
        f"Sens {val_metrics[2]:.2f} | Spec {val_metrics[3]:.2f}\n"
    )

utils.plot_loss_from_tensorboard(
    best_lost,
    train_log_dir=hps.model_dir,
    val_log_dir=os.path.join(hps.model_dir, "eval"), save_path=f"{model_dir}/result_loss.png"
)

import glob
patterns = ['pool_*.pth']
for pattern in patterns:
    search_pattern = os.path.join(model_dir, pattern)
    files = glob.glob(search_pattern)
    
    for file_path in files:
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
print(f"Saved In: {model_dir}, Accuracy: {val_metrics[0]:.2f}")