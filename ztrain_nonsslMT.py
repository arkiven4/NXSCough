import os, json, pickle, argparse, warnings, inspect

# === Scientific / Data Processing ===
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
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

df_train = pd.read_csv(f'{hps.data.db_path}/{hps.data.metadata_csv}.train')
df_test = pd.read_csv(f'{hps.data.db_path}/{hps.data.metadata_csv}.test')

class_frequencies = df_train['disease_label'].value_counts().to_dict()
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
logger.info(f"✨ Padding Type: {hps.data.pad_types}")
logger.info(f"✨ Using Model: {hps.model.pooling_model}")
logger.info(f"======================================")

epoch_str = 1
global_step = 0

pool_net = getattr(models, hps.model.pooling_model)
pool_model = pool_net(hps.model.feature_dim, **hps.model).cuda()

optimizer_p = torch.optim.AdamW(pool_model.parameters(), hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)
scheduler_p = torch.optim.lr_scheduler.ExponentialLR(optimizer_p, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)

class_code_pool_net = inspect.getsource(pool_net)
with open(f'{hps.model_dir}/model_net.py.bak', 'w') as f:
    f.write("import torch\nimport torch.nn as nn\n\n")
    f.write(class_code_pool_net)

# =============================================================
# SECTION: Additional Setup
# =============================================================
if hps.model.pooling_model.split("_")[0] == "WavLM":
    print("Loaded Pretrained WavLM")
    ssl_model = AutoModel.from_pretrained("microsoft/wavlm-large")
    pool_model.feature_extractor.load_state_dict(ssl_model.feature_extractor.state_dict())
    pool_model.feature_extractor._freeze_parameters()
    del ssl_model
elif hps.model.pooling_model.split("_")[0] == "Whisper":
    print("Loaded Pretrained Whisper")
    ssl_model = AutoModel.from_pretrained("openai/whisper-large-v3") 
    pool_model.feature_extractor.conv1.load_state_dict(ssl_model.encoder.conv1.state_dict())
    pool_model.feature_extractor.conv2.load_state_dict(ssl_model.encoder.conv2.state_dict())
    #pool_model.feature_extractor.embed_positions.load_state_dict(ssl_model.encoder.embed_positions.state_dict())
    pool_model.feature_extractor._freeze_parameters()
    del ssl_model
# =============================================================
# SECTION: Setup Logger, Dataloader
# =============================================================
best_lost = np.inf
patience_val = []

if hps.train.warm_start:
    if hps.train.from_pretrain:
        print(hps.train.warm_start_checkpoint_pool)
        checkpoint = torch.load(hps.train.warm_start_checkpoint_pool, map_location='cpu', weights_only=True)['model']
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        model_dict = {}

        for key, value in state_dict.items():
            if key.startswith('v.'):
                new_key = key[2:]
                model_dict[new_key] = value

        if hasattr(pool_model, 'module'):
            pool_model.module.load_state_dict(model_dict, strict=True)
        else:
            pool_model.load_state_dict(model_dict, strict=True)
    else:
        pool_model = utils.warm_start_model(hps.train.warm_start_checkpoint_pool, pool_model, hps.train.ignored_layer)
else:
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
    #train_loader.batch_sampler.set_epoch(epoch)
    pool_model.train()

    batch_cnt = 0
    for batch_idx, (wav_names, audio, attention_masks, dse_ids, classify_labels, regr_labels) in enumerate(tqdm(train_loader)):
        audio = audio.cuda(non_blocking=True).float().squeeze(1)
        attention_masks = attention_masks.cuda(non_blocking=True).float()
        dse_ids = dse_ids.cuda(non_blocking=True).long()
        
        with torch.amp.autocast("cuda", enabled=True):
            x_lengths = torch.tensor(commons.compute_length_from_mask(attention_masks)).cuda(non_blocking=True)

            temp_loss = []
            out_model = pool_model(audio)[0]
            temp_loss.append(utils.many_loss_category(out_model[0], dse_ids, loss_type=hps.train.loss_function, weights=class_weights_tensor, model=pool_model)[0])
            temp_loss.append(utils.many_loss_category(out_model[1], dse_ids, loss_type="CE")[0])
            for i, now_label in enumerate(classify_labels):
                if i == 0:
                    continue
                if i == len(classify_labels) - 1:
                    loss = utils.many_loss_category(out_model[i + 1], now_label.cuda(non_blocking=True).long(), loss_type="CE")[0]
                else:
                    loss = utils.many_loss_category(out_model[i + 1], now_label.cuda(non_blocking=True).long(), loss_type="BCE")[0]
                temp_loss.append(loss)

        loss_gs = temp_loss
        loss_g = sum(loss_gs) / ACCUMULATION_STEP

        scaler.scale(loss_g).backward()
        grad_norm = commons.clip_grad_value_(pool_model.parameters(), None)
        
        if (batch_cnt + 1) % ACCUMULATION_STEP == 0 or (batch_cnt + 1) == len(train_loader):
            scaler.step(optimizer_p)
            scaler.update() 
            optimizer_p.zero_grad(set_to_none=True)
        
        batch_cnt = batch_cnt + 1
        if batch_idx % hps.train.log_interval == 0:
            scalar_dict = {
                "loss/g/total": loss_g,
                "info/learning_rate": optimizer_p.param_groups[0]["lr"],
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
        for batch_idx, (wav_names, audio, attention_masks, dse_ids, classify_labels, regr_labels) in enumerate(tqdm(val_loader)):
            audio = audio.cuda(non_blocking=True).float().squeeze(1)
            attention_masks = attention_masks.cuda(non_blocking=True).float()
            dse_ids = dse_ids.cuda(non_blocking=True).long()

            temp_loss = []
            out_model = pool_model(audio)[0]
            temp_loss.append(utils.many_loss_category(out_model[0], dse_ids, loss_type=hps.train.loss_function, weights=class_weights_tensor, model=pool_model)[0])
            temp_loss.append(utils.many_loss_category(out_model[1], dse_ids, loss_type="CE")[0])
            for i, now_label in enumerate(classify_labels):
                if i == 0:
                    continue
                if i == len(classify_labels) - 1:
                    loss = utils.many_loss_category(out_model[i + 1], now_label.cuda(non_blocking=True).long(), loss_type="CE")[0]
                else:
                    loss = utils.many_loss_category(out_model[i + 1], now_label.cuda(non_blocking=True).long(), loss_type="BCE")[0]
                temp_loss.append(loss)

            loss_gs = temp_loss
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
    scheduler_p.step()

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

        if epoch > 3 and len(patience_val) >= 4:
            new_lr = hps.train.learning_rate * (0.9 ** ((epoch - 3) // 1))
            for param_group in optimizer_p.param_groups:
                param_group['lr'] = new_lr

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


######################################################################
#
# Metric Best Model
#
######################################################################

_, _, _, _, epoch_str = utils.load_checkpoint(
    os.path.join(hps.model_dir, "best_pool.pth"),
    pool_model,
    optimizer_p,
    scheduler_p,
)

pool_model.eval() 
all_preds, all_labels, all_probs  = [], [], []
with torch.no_grad():
    for batch_idx, (wav_names, audio, attention_masks, dse_ids, classify_labels, regr_labels) in enumerate(tqdm(train_loader)):
        audio = audio.cuda(non_blocking=True).float().squeeze(1)
        attention_masks = attention_masks.cuda(non_blocking=True).float()
        dse_ids = dse_ids.cuda(non_blocking=True).long()

        out_model = pool_model(audio)[0]
        outputs = out_model[0]
        
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(dse_ids.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

n_classes = len(CLASS_NAMES)
all_probs = np.array(all_probs)
all_labels = np.array(all_labels)
fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1])
train_auc_score = auc(fpr, tpr)

cm = confusion_matrix(all_labels, all_preds)
train_accuracy = accuracy_score(all_labels, all_preds, normalize=True)
train_b_accuracy = balanced_accuracy_score(all_labels, all_preds)
train_f1 = f1_score(all_labels, all_preds, average="weighted")
train_f1_pos = f1_score(all_labels, all_preds, pos_label=1)
try:
    train_roc_auc = roc_auc_score(all_labels, all_preds)
except Exception as exception:
    train_roc_auc = None
tn, fp, fn, tp = cm.ravel()
train_sensitivity = tp / (tp + fn)
train_specificity = tn / (tn + fp)

########

all_preds, all_labels, all_probs  = [], [], []
with torch.no_grad():
    for batch_idx, (wav_names, audio, attention_masks, dse_ids, classify_labels, regr_labels) in enumerate(tqdm(val_loader)):
        audio = audio.cuda(non_blocking=True).float().squeeze(1)
        attention_masks = attention_masks.cuda(non_blocking=True).float()
        dse_ids = dse_ids.cuda(non_blocking=True).long()

        out_model = pool_model(audio)[0]
        outputs = out_model[0]
        
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(dse_ids.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig(f"{model_dir}/result_cm.png")

# --- ROC Curve ---
n_classes = len(CLASS_NAMES)
all_probs = np.array(all_probs)
all_labels = np.array(all_labels)
fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1])
auc_score = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}", color="darkorange")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig(f"{model_dir}/result_roc.png")

accuracy = accuracy_score(all_labels, all_preds, normalize=True)
b_accuracy = balanced_accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average="weighted")
f1_pos = f1_score(all_labels, all_preds, pos_label=1)
try:
    roc_auc = roc_auc_score(all_labels, all_preds)
except Exception as exception:
    roc_auc = None

tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

with open(f"{model_dir}/result_summary.txt", "w") as file:
    file.write(df['disease_label'].value_counts().to_string() + "\n")
    file.write("\n")
    file.write(f"Accuracy {train_accuracy:.2f} | Balanced Accuracy {train_b_accuracy:.2f} | AUC {train_auc_score:.2f} | ROC AUC {train_roc_auc:.2f} | Weighted F1: {train_f1:.2f} | Positive F1: {train_f1_pos:.2f} | Sensitivity: {train_sensitivity:.2f} | Specificity: {train_specificity:.2f} \n")
    file.write(f"Accuracy {accuracy:.2f} | Balanced Accuracy {b_accuracy:.2f} | AUC {auc_score:.2f} | ROC AUC {roc_auc:.2f} | Weighted F1: {f1:.2f} | Positive F1: {f1_pos:.2f} | Sensitivity: {sensitivity:.2f} | Specificity: {specificity:.2f} \n")

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

print(f"Saved In: {model_dir}, Accuracy: {accuracy:.2f}, AUC: {auc_score:.2f}")