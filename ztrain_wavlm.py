import os, json, pickle
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split

from transformers import AutoModel, Wav2Vec2FeatureExtractor, WavLMForXVector
import utils
import commons
import models
import models_pooling
from cough_datasets import SERDatasets, SERDatasetsCollate

import warnings
warnings.simplefilter("ignore", UserWarning)

# =============================================================
# SECTION: Intialize Data
# =============================================================

MODEL_NAME = "Cat_PoolingSep_512_Roberto_normmax"
CONFIG_PATH = "configs/emoclassicat.json"

model_dir = os.path.join("./logs", MODEL_NAME)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

config_save_path = os.path.join(model_dir, "config.json")
if True:
    with open(CONFIG_PATH, "r") as f:
      data = f.read()
    with open(config_save_path, "w") as f:
      f.write(data)

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
emotion_codes = [0, 1]

df = pd.read_csv(f'{hps.data.db_path}/{hps.data.metadata_csv}')
df = df[df['database'].isin(['tb_longitudinal_data', 'tb_solicited_data'])]

df_train, df_test = train_test_split(df, test_size=0.03, random_state=42, shuffle=True)
# df_train = df[df['database'].isin(['tb_longitudinal_data'])]
# df_test = df[df['database'].isin(['tb_solicited_data'])]

class_frequencies = df_train['disease_label'].value_counts().to_dict()
total_samples = len(df_train)
class_weights = {cls: total_samples / (len(emotion_codes) * freq) if freq != 0 else 0 for cls, freq in class_frequencies.items()}
weights_list = [class_weights[cls] for cls in emotion_codes]
class_weights_tensor = torch.tensor(weights_list, device='cuda', dtype=torch.float)

df_train.drop(['database'], axis=1, inplace=True)
df_test.drop(['database'], axis=1, inplace=True)

# =============================================================
# SECTION: Setup Logger, Dataloader
# =============================================================
logger = utils.get_logger(hps.model_dir)
logger.info(hps)

writer = SummaryWriter(log_dir=hps.model_dir)
writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

collate_fn = SERDatasetsCollate(1)
train_dataset = SERDatasets(df_train.values, hps.data)
val_dataset = SERDatasets(df_test.values, hps.data)

#train_sampler = DistributedBucketSampler(train_dataset, cur_bs, [32,300,400,500,600,700,800,900,1000], num_replicas=1, rank=0, shuffle=True)
#train_loader = DataLoader(train_dataset, num_workers=28, shuffle=False, pin_memory=True, collate_fn=collate_fn, batch_sampler=train_sampler)
train_loader = DataLoader(train_dataset, num_workers=28, shuffle=True, batch_size=cur_bs, pin_memory=True, drop_last=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, num_workers=28, shuffle=False, batch_size=hps.train.batch_size, pin_memory=True, drop_last=True, collate_fn=collate_fn)

# =============================================================
# SECTION: Setup Logger, Dataloader
# =============================================================
epoch_str = 1
global_step = 0

ssl_model = AutoModel.from_pretrained("microsoft/wavlm-large")
ssl_model.freeze_feature_encoder()
ssl_model.eval(); ssl_model.cuda()
ssl_feat_dim = ssl_model.config.hidden_size

pool_net = getattr(models_pooling, hps.model.pooling_type)
pool_model = pool_net(ssl_feat_dim, **hps.model).cuda()
reg_model = models.HeadCatPrediction(**hps.model).cuda()

optimizer_ssl = torch.optim.AdamW(ssl_model.parameters(), 1e-5)
optimizer_p = torch.optim.AdamW(pool_model.parameters(), hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)
optimizer_g = torch.optim.AdamW(reg_model.parameters(), hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)

scheduler_ssl = torch.optim.lr_scheduler.ExponentialLR(optimizer_ssl, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)
scheduler_p = torch.optim.lr_scheduler.ExponentialLR(optimizer_p, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)
scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optimizer_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)

# =============================================================
# SECTION: Setup Logger, Dataloader
# =============================================================
best_lost = np.inf
patience_val = []

if hps.train.warm_start:
    ssl_model = utils.warm_start_model(hps.train.warm_start_checkpoint_ssl, ssl_model, hps.train.ignored_layer)
    pool_model = utils.warm_start_model(hps.train.warm_start_checkpoint_pool, pool_model, hps.train.ignored_layer)
    reg_model = utils.warm_start_model(hps.train.warm_start_checkpoint_reg, reg_model, hps.train.ignored_layer)
else:
    try:
        _, _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "ssl_*.pth"),
            ssl_model,
            optimizer_ssl,
            scheduler_ssl,
        )
        _, _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "pool_*.pth"),
            pool_model,
            optimizer_p,
            scheduler_p,
        )
        _, _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "reg_*.pth"),
            reg_model,
            optimizer_g,
            scheduler_g,
        )

        epoch_str += 1
        global_step = (epoch_str - 1) * len(train_loader)

        with open(os.path.join(hps.model_dir, "traindata.pickle"), 'rb') as handle:
            traindata = pickle.load(handle)
            best_lost = traindata['best_lost']
            patience_val = traindata['patience_val']
        
    except Exception as e:
        print(e)

scaler = GradScaler()

optimizer_ssl.zero_grad(set_to_none=True)
optimizer_p.zero_grad(set_to_none=True)
optimizer_g.zero_grad(set_to_none=True)

# =============================================================
# SECTION: Train Epoch
# =============================================================

for epoch in range(epoch_str, hps.train.epochs + 1):
    #train_loader.batch_sampler.set_epoch(epoch)
    ssl_model.train()
    pool_model.train()
    reg_model.train()

    batch_cnt = 0

    for batch_idx, (wav_names, audio, attention_masks, dse_id) in enumerate(tqdm(train_loader)):
        audio = audio.cuda(non_blocking=True).float().squeeze(1)
        attention_masks = attention_masks.cuda(non_blocking=True).float()
        dse_id = dse_id.cuda(non_blocking=True).long()
        
        with torch.amp.autocast("cuda", enabled=True):
            ssl_hidden = ssl_model(audio, attention_mask=attention_masks, output_hidden_states=hps.model.output_hidden_states)
            if hps.model.output_hidden_states:
                ssl_hidden = ssl_hidden[2]
                ssl_hidden = torch.stack(ssl_hidden, dim=1)
            else:
                ssl_hidden = ssl_hidden.last_hidden_state
            
            x_lengths = torch.tensor(commons.compute_length_from_mask(attention_masks)).cuda(non_blocking=True)
            ssl_hidden = pool_model(ssl_hidden, x_lengths, attention_mask=attention_masks)
            dse_pred = reg_model(ssl_hidden)

            loss, f1_micro, f1_macro, accuracy = utils.CE_weight_category(dse_pred, dse_id, class_weights_tensor)

        loss_gs = [loss]
        loss_g = sum(loss_gs) / ACCUMULATION_STEP

        scaler.scale(loss_g).backward()
        grad_norm1 = commons.clip_grad_value_(ssl_model.parameters(), None)
        grad_norm2 = commons.clip_grad_value_(pool_model.parameters(), None)
        grad_norm3 = commons.clip_grad_value_(reg_model.parameters(), None)
        
        if (batch_cnt + 1) % ACCUMULATION_STEP == 0 or (batch_cnt + 1) == len(train_loader):
            scaler.step(optimizer_ssl)
            scaler.step(optimizer_p)
            scaler.step(optimizer_g)
            scaler.update() 
            optimizer_ssl.zero_grad(set_to_none=True)
            optimizer_p.zero_grad(set_to_none=True)
            optimizer_g.zero_grad(set_to_none=True)
        
        batch_cnt = batch_cnt + 1

        if batch_idx % hps.train.log_interval == 0:
            scalar_dict = {
                "loss/g/total": loss_g,
                "loss/g/f1_micro": f1_micro,
                "loss/g/f1_macro": f1_macro,
                "loss/g/accuracy": accuracy,
                "learning_rate/1": optimizer_ssl.param_groups[0]["lr"],
                "learning_rate/2": optimizer_p.param_groups[0]["lr"],
                "learning_rate/3": optimizer_g.param_groups[0]["lr"],
                "grad_norm/1": grad_norm1,
                "grad_norm/2": grad_norm2,
                "grad_norm/3": grad_norm3,
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
        
    ssl_model.eval()
    pool_model.eval()
    reg_model.eval()
    losses_tot = []
    acc_tot = []
    with torch.no_grad():
        for batch_idx, (wav_names, audio, attention_masks, dse_ids) in enumerate(tqdm(val_loader)):
            audio = audio.cuda(non_blocking=True).float().squeeze(1)
            attention_masks = attention_masks.cuda(non_blocking=True).float()
            dse_ids = dse_ids.cuda(non_blocking=True).long()
            
            ssl_hidden = ssl_model(audio, attention_mask=attention_masks, output_hidden_states=hps.model.output_hidden_states)
            if hps.model.output_hidden_states:
                ssl_hidden = ssl_hidden[2]
                ssl_hidden = torch.stack(ssl_hidden, dim=1)
            else:
                ssl_hidden = ssl_hidden.last_hidden_state            
            
            x_lengths = torch.tensor(commons.compute_length_from_mask(attention_masks)).cuda(non_blocking=True).long()
            ssl_hidden = pool_model(ssl_hidden, x_lengths, attention_mask=attention_masks)
            dse_pred = reg_model(ssl_hidden)

            loss, f1_micro, f1_macro, accuracy = utils.CE_weight_category(dse_pred, dse_ids, class_weights_tensor)

            loss_gs = [loss]
            loss_g = sum(loss_gs)

            if batch_idx == 0:
                losses_tot = loss_gs
                acc_tot = [accuracy]
            else:
                losses_tot = [x + y for (x, y) in zip(losses_tot, loss_gs)]
                acc_tot = [x + y for (x, y) in zip(acc_tot, [accuracy])]

    losses_tot = [x / len(val_loader) for x in losses_tot]
    acc_tot = [x / len(val_loader) for x in acc_tot]
    loss_tot = torch.mean(torch.tensor(losses_tot)) #sum(losses_tot)
    acc_tot = torch.mean(torch.tensor(acc_tot)) #sum(losses_tot)
    scalar_dict = {"loss/g/total": loss_tot, "ACC": acc_tot}
    scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_tot)})
    utils.summarize(
        writer=writer_eval, global_step=global_step, scalars=scalar_dict
    )

    ssl_model.train()
    pool_model.train()
    reg_model.train()

    #scheduler_ssl.step()
    #scheduler_p.step()
    #scheduler_g.step()

    if loss_tot < best_lost and loss_tot > 0:
        logger.info(f"Get Best New Validation!!!!")
        best_lost = loss_tot
        patience_val = []

        utils.save_checkpoint(
            ssl_model, optimizer_ssl, scheduler_ssl,
            hps.train.learning_rate, epoch,
            os.path.join(hps.model_dir, "best_ssl.pth"))
        utils.save_checkpoint(
            pool_model, optimizer_p, scheduler_p,
            hps.train.learning_rate, epoch,
            os.path.join(hps.model_dir, "best_pool.pth"))
        utils.save_checkpoint(
            reg_model, optimizer_g, scheduler_g,
            hps.train.learning_rate, epoch,
            os.path.join(hps.model_dir, "best_reg.pth"))
    else:
        patience_val.append(1)
        logger.info(f"Patience: {len(patience_val)}")

        if epoch > 3 and len(patience_val) >= 4:
            for param_group in optimizer_ssl.param_groups:
                param_group['lr'] = 1e-5 * (0.98 ** ((epoch - 3) // 1))

            new_lr = hps.train.learning_rate * (0.9 ** ((epoch - 3) // 1))
            for param_group in optimizer_p.param_groups:
                param_group['lr'] = new_lr
            for param_group in optimizer_g.param_groups:
                param_group['lr'] = new_lr

        if len(patience_val) > 15:
            break


    logger.info("====> Epoch: {}".format(epoch))
    if epoch % 1 == 0:
        utils.save_checkpoint(
            ssl_model, optimizer_ssl, scheduler_ssl,
            hps.train.learning_rate, epoch,
            os.path.join(hps.model_dir, "ssl_{}.pth".format(epoch)))
        utils.save_checkpoint(
            pool_model, optimizer_p, scheduler_p,
            hps.train.learning_rate, epoch,
            os.path.join(hps.model_dir, "pool_{}.pth".format(epoch)))
        utils.save_checkpoint(
            reg_model, optimizer_g, scheduler_g,
            hps.train.learning_rate, epoch,
            os.path.join(hps.model_dir, "reg_{}.pth".format(epoch)))

        with open(os.path.join(hps.model_dir, "traindata.pickle"), 'wb') as handle:
            pickle.dump({
                "best_lost": best_lost,
                "patience_val": patience_val
            }, handle, protocol=pickle.HIGHEST_PROTOCOL)