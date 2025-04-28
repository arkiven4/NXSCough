import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from tqdm import tqdm
import csv

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

from transformers import AutoModel, Wav2Vec2FeatureExtractor, WavLMForXVector

import utils
import models
import models_pooling
import commons
from dataset import SERDatasets, SERSubmitDatasets, SERDatasetsCollate, SERSubmitDatasetsCollate, DistributedBucketSampler
from layers import MultiheadAttentionPooling

model_name = "Cat_PoolingSep_512_Roberto_normmax"
chkpt_index = 11
is_Best = True
is_Eval = True

# =============================================================
# SECTION: Load Model
# =============================================================
best_prefix = "best_" if is_Best else ""
numeric_sufix = "" if is_Best else f"_{chkpt_index}"

model_dir = f"logs/{model_name}"
hps = utils.get_hparams_from_dir(model_dir)

ssl_model = AutoModel.from_pretrained("microsoft/wavlm-large")
ssl_model.freeze_feature_encoder()
utils.load_checkpoint(f"{model_dir}/{best_prefix}ssl{numeric_sufix}.pth", ssl_model)
ssl_model.eval(); ssl_model.cuda()
ssl_feat_dim = ssl_model.config.hidden_size

pool_net = getattr(models_pooling, hps.model.pooling_type)
pool_model = pool_net(ssl_feat_dim, **hps.model).cuda()
utils.load_checkpoint(f"{model_dir}/{best_prefix}pool{numeric_sufix}.pth", pool_model)
pool_model.eval()

reg_model = models.EmoDimPrediction(**hps.model).cuda()
utils.load_checkpoint(f"{model_dir}/{best_prefix}reg{numeric_sufix}.pth", reg_model)
reg_model.eval()
print("Loading Models")

# =============================================================
# SECTION: Setup Data
# =============================================================

if is_Eval:
    emotions = ["Angry", "Sad", "Happy", "Surprise", "Fear", "Disgust", "Contempt", "Neutral"]
    emotion_codes = ["A", "S", "H", "U", "F", "D", "C", "N"]

    df = pd.read_csv('../Data_Raw/Labels/labels_isoforest.csv')
    df = df[df['EmoClass'].isin(emotion_codes)]
    df['EmoClassCode'] = df['EmoClass'].apply(lambda x: emotion_codes.index(x))
    temp_speakerlist = list(df['SpkrID'].unique())
    df['SpkrID']  = df['SpkrID'].apply(lambda x: temp_speakerlist.index(x))
    speaker_number = list(set(df['SpkrID'].unique()))[-1] + 1

    df_train = df[df['Split_Set'] == 'Train'].copy()
    class_frequencies = df_train['EmoClass'].value_counts().to_dict()
    total_samples = len(df_train)
    class_weights = {cls: total_samples / (len(emotion_codes) * freq) if freq != 0 else 0 for cls, freq in class_frequencies.items()}
    weights_list = [class_weights[cls] for cls in emotion_codes]
    class_weights_tensor = torch.tensor(weights_list, device='cuda', dtype=torch.float)

    df = df[df['Split_Set'] == 'Development'].copy()
    df.reset_index(drop=True, inplace=True)

    df.drop(['EmoClass', 'Gender', 'Split_Set'], axis=1, inplace=True)
    df.reset_index(drop=True, inplace=True)

    collate_fn = SERDatasetsCollate(1)
    val_dataset = SERDatasets(df.values, hps.data)
    val_loader = DataLoader(val_dataset, num_workers=28, shuffle=False, batch_size=hps.train.batch_size, 
                            pin_memory=True, drop_last=True, collate_fn=collate_fn)

    emo_id_array = []
    emo_pred_array = []
    with torch.no_grad():
        for batch_idx, (wav_names, audio, attention_masks, spk_emb, txt_emb, txt_masks, emo_dim, spk_id, emo_id) in enumerate(tqdm(val_loader)):
            audio = audio.cuda(non_blocking=True).float().squeeze(1)
            attention_masks = attention_masks.cuda(non_blocking=True).float()
            txt_masks = txt_masks.cuda(non_blocking=True).long()
            spk_emb = spk_emb.cuda(non_blocking=True).float()
            emo_id = emo_id.cuda(non_blocking=True).float()
            txt_emb = txt_emb.cuda(non_blocking=True).float()
            
            ssl_hidden = ssl_model(audio, attention_mask=attention_masks, output_hidden_states=hps.model.output_hidden_states)
            if hps.model.output_hidden_states:
                ssl_hidden = ssl_hidden[2]
                ssl_hidden = torch.stack(ssl_hidden, dim=1)
            else:
                ssl_hidden = ssl_hidden.last_hidden_state            
            
            x_lengths = torch.tensor(commons.compute_length_from_mask(attention_masks)).cuda(non_blocking=True).long()
            ssl_hidden = pool_model(ssl_hidden, x_lengths, spk_embeds=spk_emb, txt_embeds=txt_emb, attention_mask=attention_masks, txt_masks=txt_masks)
            emo_pred = reg_model(ssl_hidden)
            
            emo_id_array.append(emo_id.cpu().detach().numpy())
            emo_pred_array.append(np.argmax(emo_pred.cpu().detach().numpy(), axis=-1))

    emo_id_array = torch.from_numpy(np.hstack(emo_id_array).reshape(-1))
    emo_pred_array = torch.from_numpy(np.hstack(emo_pred_array).reshape(-1))

    f1_micro, f1_macro, accuracy = utils.CE_weight_category(emo_pred_array, emo_id_array, class_weights_tensor, test=True)
    print(f1_micro)
    print(f1_macro)
else:
    audio_path = "../Data_Raw/Audios"
    files_test3 = [filename for filename in os.listdir(audio_path) if 'test3' in filename]

    collate_fn_test = SERSubmitDatasetsCollate(1)
    test_dataset = SERSubmitDatasets(np.array(files_test3).reshape(-1, 1), hps.data)
    test_loader = DataLoader(test_dataset, num_workers=28, shuffle=False, batch_size=16, 
                            pin_memory=True, drop_last=True, collate_fn=collate_fn_test)

    total_wav_name = []
    total_emo_cat = []
    with torch.no_grad():
        for batch_idx, (audio_name, audio, attention_masks, spk_emb) in enumerate(tqdm(test_loader)):
            audio = audio.cuda(non_blocking=True).float().squeeze(1)
            attention_masks = attention_masks.cuda(non_blocking=True).float()
            spk_emb = spk_emb.cuda(non_blocking=True).float()
            
            ssl_hidden = ssl_model(audio, attention_mask=attention_masks, output_hidden_states=hps.model.output_hidden_states)
            if hps.model.output_hidden_states:
                ssl_hidden = ssl_hidden[2]
                ssl_hidden = torch.stack(ssl_hidden, dim=1)
            else:
                ssl_hidden = ssl_hidden.last_hidden_state

            x_lengths = torch.tensor(commons.compute_length_from_mask(attention_masks)).cuda(non_blocking=True).long()
            emo_pred = model(ssl_hidden, x_lengths, spk_embeds=spk_emb, attention_mask=attention_masks) # emo_act, emo_val, emo_dom

            total_wav_name = total_wav_name + audio_name
            total_emo_cat = total_emo_cat + np.argmax(emo_pred.cpu().detach().numpy(), axis=-1).tolist()

    datato_Submit = []
    for audio_name, emo_prd in zip(total_wav_name, total_emo_cat):
        emo_prd = emotion_codes[emo_prd]
        datato_Submit.append([audio_name, emo_prd])

    os.makedirs(model_dir + '/results', exist_ok=True) 
    csv_filename = model_dir + '/results/' + "test3_cls" + '.csv'
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["FileName", "EmoClass"])
        writer.writerows(datato_Submit)

    df_tosbumit = pd.read_csv(csv_filename)
    df_tosbumit = df_tosbumit.sort_values(by='FileName') 
    df_tosbumit.reset_index(drop=True, inplace=True)
    df_tosbumit.to_csv(csv_filename, index=False)
