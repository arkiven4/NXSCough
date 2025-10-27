import os, json, pickle, argparse, warnings, inspect, subprocess, socket

# === Scientific / Data Processing ===
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_curve,
    auc,
    roc_auc_score,
    f1_score,
)

# === PyTorch Core ===
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler

# === Transformers / PEFT ===
from transformers import AutoModel, AutoConfig, AutoFeatureExtractor
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# === TensorBoard Utility ===
from tensorboard.backend.event_processing import event_accumulator

# === Project-Specific Modules ===
import utils
import commons
import models
from cough_datasets import (
    MTCoughDatasets,
    MTCoughDatasetsCollate,
    CoughDatasets,
    CoughDatasetsCollate,
)

# === Suppress Warnings ===
warnings.simplefilter("ignore", UserWarning)

def get_free_port():
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port

def stratified_group_split(df, label_col='tb_status', group_col='participant', test_size=0.2, random_state=42):
    sgkf = StratifiedGroupKFold(n_splits=int(1/test_size), shuffle=True, random_state=random_state)
    for train_idx, val_idx in sgkf.split(df, df[label_col], df[group_col]):
        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_val = df.iloc[val_idx].reset_index(drop=True)
        return df_train, df_val

# =============================================================
# SECTION: Intialize Data
# =============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--init", action="store_true")
parser.add_argument("--model_name", type=str, default="try_wavlmlora_downstream")
parser.add_argument("--config_path", type=str, default="configs/ssl_finetuning.json")
args = parser.parse_args()

model_dir = os.path.join("./logs", args.model_name)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
port = get_free_port()
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
##### Label Umum Semua
Diseases_codes = [0, 1]
CLASS_NAMES = ["Healthy", "TB"]

df_longi = pd.read_csv('/run/media/fourier/Data1/Pras/DatabaseLLM/coda/longitudinal_original.csv')
df_solic = pd.read_csv('/run/media/fourier/Data1/Pras/DatabaseLLM/coda/solicited_original.csv')

participant_mapping_longi = {participant: idx for idx, participant in enumerate(set(np.concatenate([df_solic['participant'].unique(), df_longi['participant'].unique()])))}
df_longi['participant'] = df_longi['participant'].map(participant_mapping_longi)
df_solic['participant'] = df_solic['participant'].map(participant_mapping_longi)

gender_mapping_longi = {gender: idx for idx, gender in enumerate(df_longi['sex'].unique())}
df_longi['sex'] = df_longi['sex'].map(gender_mapping_longi)
df_solic['sex'] = df_solic['sex'].map(gender_mapping_longi)

df_longi_train, df_longi_val = stratified_group_split(df_longi)
df_solic_train, df_solic_val = stratified_group_split(df_solic)

df_train = pd.concat([df_longi_train, df_solic_train], ignore_index=True)
df_test = pd.concat([df_longi_val, df_solic_val], ignore_index=True)

if hps.data.reorder_target:
    cols = hps.data.column_order
    df_train = df_train[cols]
    df_test = df_test[cols]

class_frequencies = df_train[hps.data.target_column].value_counts().to_dict()
total_samples = len(df_train)
class_weights = {cls: total_samples / (len(Diseases_codes) * freq) if freq != 0 else 0 for cls, freq in class_frequencies.items()}
weights_list = [class_weights[cls] for cls in Diseases_codes]
class_weights_tensor = torch.tensor(weights_list, device='cuda', dtype=torch.float)
class_weights_tensor = None
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

#train_sampler = DistributedBucketSampler(train_dataset, cur_bs, [32,300,400,500,600,700,800,900,1000], num_replicas=1, rank=0, shuffle=True)
#train_loader = DataLoader(train_dataset, num_workers=28, shuffle=False, pin_memory=True, collate_fn=collate_fn, batch_sampler=train_sampler)
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
logger.info(f"✨ Tensorboard: {hps.model.pooling_model}")
logger.info(f"TensorBoard link: http://100.101.198.75:{port}")
logger.info(f"======================================")

epoch_str = 1
global_step = 0

ssl_model = AutoModel.from_pretrained("microsoft/wavlm-large") # openai/whisper-large-v3 # microsoft/wavlm-large
#ssl_model.freeze_feature_encoder()
#ssl_model.eval(); ssl_model.cuda()
ssl_feat_dim = ssl_model.config.hidden_size

# config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")
# ssl_model = get_peft_model(ssl_model, config)
# ssl_model.print_trainable_parameters()
ssl_model.cuda()

pool_net = getattr(models, hps.model.pooling_model)
pool_model = pool_net(ssl_feat_dim, **hps.model).cuda()

optimizer_ssl = torch.optim.AdamW(ssl_model.parameters(), lr=1e-5)
optimizer_p = torch.optim.AdamW(pool_model.parameters(), hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)

scheduler_ssl = torch.optim.lr_scheduler.ExponentialLR(optimizer_ssl, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)
scheduler_p = torch.optim.lr_scheduler.ExponentialLR(optimizer_p, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)

class_code_pool_net = inspect.getsource(pool_net)
with open(f'{hps.model_dir}/model_net.py.bak', 'w') as f:
    f.write("import torch\nimport torch.nn as nn\n\n")
    f.write(class_code_pool_net)

# =============================================================
# SECTION: Setup Logger, Dataloader
# =============================================================
best_lost = np.inf
patience_val = []

scaler = GradScaler('cuda')
optimizer_ssl.zero_grad(set_to_none=True)
optimizer_p.zero_grad(set_to_none=True)

# =============================================================
# SECTION: Train Epoch
# =============================================================
import torch.nn.functional as F

def orthogonality_loss(d_emb, s_emb):
    d = F.normalize(d_emb, p=2, dim=1)
    s = F.normalize(s_emb, p=2, dim=1)
    dots = torch.sum(d * s, dim=1)  # (B,)
    return torch.mean(dots * dots)

for epoch in range(epoch_str, hps.train.epochs + 1):
    pool_model.train()

    batch_cnt = 0
    for batch_idx, (wav_names, audio, attention_masks, dse_ids, othr_ids) in enumerate(tqdm(train_loader)):
        audio = audio.cuda(non_blocking=True).float().squeeze(1)
        attention_masks = attention_masks.cuda(non_blocking=True).float()
        dse_ids = dse_ids.cuda(non_blocking=True).float()
        spk_ids = othr_ids[0].cuda(non_blocking=True).long()
        gndr_ids = othr_ids[1].cuda(non_blocking=True).long()
        
        with torch.amp.autocast("cuda", enabled=True):
            x_lengths = torch.tensor(commons.compute_length_from_mask(attention_masks)).cuda(non_blocking=True)
            out_model = pool_model(audio, grl_lambda=1.0)

        ld = utils.many_loss_category(out_model["disease_logits"], dse_ids, loss_type=hps.train.loss_function, weights=class_weights_tensor, model=pool_model)
        ls = utils.many_loss_category(out_model["speaker_logits"], spk_ids, loss_type=hps.train.loss_function, model=pool_model)[0] * 0.1
        lg = utils.many_loss_category(out_model["gender_logits"], gndr_ids, loss_type=hps.train.loss_function, model=pool_model)[0] * 0.1
        l_ortho = orthogonality_loss(out_model["d_emb"], out_model["s_emb"]) #* 0.3
        loss = ld + [ls] + [lg] + [l_ortho]

        loss_gs = loss
        loss_g = sum(loss_gs) / ACCUMULATION_STEP

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

        global_step += 1
        
    pool_model.eval()
    losses_tot = []
    acc_tot = []
    with torch.no_grad():
        for batch_idx, (wav_names, audio, attention_masks, dse_ids, othr_ids) in enumerate(tqdm(val_loader)):
            audio = audio.cuda(non_blocking=True).float().squeeze(1)
            attention_masks = attention_masks.cuda(non_blocking=True).float()
            dse_ids = dse_ids.cuda(non_blocking=True).float()
            spk_ids = othr_ids[0].cuda(non_blocking=True).long()
            gndr_ids = othr_ids[1].cuda(non_blocking=True).long()

            x_lengths = torch.tensor(commons.compute_length_from_mask(attention_masks)).cuda(non_blocking=True).long()
            
            out_model = pool_model(audio, grl_lambda=0.0)
            loss = utils.many_loss_category(out_model["disease_logits"], dse_ids, loss_type=hps.train.loss_function, weights=class_weights_tensor, model=pool_model)
            #loss, f1_micro, f1_macro, accuracy = utils.CE_weight_category(dse_pred, dse_ids, class_weights_tensor)

            loss_gs = loss
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

        # if epoch > 3 and len(patience_val) >= 4:
        #     new_lr = hps.train.learning_rate * (0.9 ** ((epoch - 3) // 1))
        #     for param_group in optimizer_p.param_groups:
        #         param_group['lr'] = new_lr

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
# df_solic = pd.read_csv('/run/media/fourier/Data1/Pras/DatabaseLLM/coda/solicited_original.csv')
# _, df_test = stratified_group_split(df_solic)
# hps.data.target_column = 'tb_status'

# if hps.data.reorder_target:
#     cols = ["path_file", hps.data.target_column]
#     df_test = df_test[cols]

val_dataset = CoughDatasets(df_test.values, hps.data, train=False)
val_loader = DataLoader(val_dataset, num_workers=28, shuffle=False, batch_size=hps.train.batch_size, pin_memory=True, drop_last=True, collate_fn=collate_fn)


_, _, _, _, epoch_str = utils.load_checkpoint(
    os.path.join(hps.model_dir, "best_pool.pth"),
    pool_model,
    None,
    None,
)

pool_model.eval() 
all_preds, all_labels, all_probs  = [], [], []
with torch.no_grad():
    for batch_idx, (wav_names, audio, attention_masks, dse_ids, othr_ids) in enumerate(tqdm(train_loader)):
        audio = audio.cuda(non_blocking=True).float().squeeze(1)
        attention_masks = attention_masks.cuda(non_blocking=True).float()
        dse_ids = dse_ids.cuda(non_blocking=True).float()

        x_lengths = torch.tensor(commons.compute_length_from_mask(attention_masks)).cuda(non_blocking=True).long()
        out_model = pool_model(audio)
        outputs = out_model["disease_logits"]
        
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(outputs, dim=1)
        dse_ids = np.argmax(dse_ids.cpu().detach().numpy(), axis=-1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(dse_ids)
        all_probs.extend(probs.cpu().numpy())

n_classes = len(CLASS_NAMES)
all_probs = np.array(all_probs)
all_labels = np.array(all_labels)

cm = confusion_matrix(all_labels, all_preds)
train_accuracy = accuracy_score(all_labels, all_preds, normalize=True)
train_b_accuracy = balanced_accuracy_score(all_labels, all_preds)
train_f1 = f1_score(all_labels, all_preds, average="weighted")
train_f1_pos = f1_score(all_labels, all_preds, average="macro")
train_roc_auc = None

# For multiclass, calculate average sensitivity and specificity
train_sensitivity = np.mean([cm[i, i] / cm[i, :].sum() for i in range(len(CLASS_NAMES)) if cm[i, :].sum() > 0])
train_specificity = np.mean([(cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]) / (cm.sum() - cm[i, :].sum()) 
                            for i in range(len(CLASS_NAMES)) if (cm.sum() - cm[i, :].sum()) > 0])

########
all_preds, all_labels, all_probs  = [], [], []
with torch.no_grad():
    for batch_idx, (wav_names, audio, attention_masks, dse_ids, othr_ids) in enumerate(tqdm(val_loader)):
        audio = audio.cuda(non_blocking=True).float().squeeze(1)
        attention_masks = attention_masks.cuda(non_blocking=True).float()
        dse_ids = dse_ids.cuda(non_blocking=True).float()

        x_lengths = torch.tensor(commons.compute_length_from_mask(attention_masks)).cuda(non_blocking=True).long()
        out_model = pool_model(audio)
        outputs = out_model["disease_logits"]
        
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(outputs, dim=1)
        dse_ids = np.argmax(dse_ids.cpu().detach().numpy(), axis=-1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(dse_ids)
        all_probs.extend(probs.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig(f"{model_dir}/result_cm.png")

# Remove ROC curve for multiclass classification
n_classes = len(CLASS_NAMES)
all_probs = np.array(all_probs)
all_labels = np.array(all_labels)

accuracy = accuracy_score(all_labels, all_preds, normalize=True)
b_accuracy = balanced_accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average="weighted")
f1_pos = f1_score(all_labels, all_preds, average="macro")
roc_auc = None

sensitivity = np.mean([cm[i, i] / cm[i, :].sum() for i in range(len(CLASS_NAMES)) if cm[i, :].sum() > 0])
specificity = np.mean([(cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]) / (cm.sum() - cm[i, :].sum()) 
                      for i in range(len(CLASS_NAMES)) if (cm.sum() - cm[i, :].sum()) > 0])

with open(f"{model_dir}/result_summary.txt", "w") as file:
    file.write(f"Train - Accuracy {train_accuracy:.2f} | Balanced Accuracy {train_b_accuracy:.2f} | ROC AUC {train_roc_auc} | Weighted F1: {train_f1:.2f} | Positive F1: {train_f1_pos:.2f} | Sensitivity: {train_sensitivity:.2f} | Specificity: {train_specificity:.2f} \n")
    file.write(f"Val - Accuracy {accuracy:.2f} | Balanced Accuracy {b_accuracy:.2f} | ROC AUC {roc_auc} | Weighted F1: {f1:.2f} | Positive F1: {f1_pos:.2f} | Sensitivity: {sensitivity:.2f} | Specificity: {specificity:.2f} \n")

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

print(f"Saved In: {model_dir}, Accuracy: {accuracy:.2f}")