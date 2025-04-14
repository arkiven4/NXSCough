import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_curve, auc, roc_auc_score, f1_score

from datasets import AudioDataset, collate_fn, df_sampler
import models
import utils

DATA_PATH = "/run/media/fourier/Data1/Pras/Database_ThesisNew/"
METADATA = "tb_solic_filt.csv" # tb_all_filt.csv
RANDOM_CODE = utils.generate_random_code()
OUTPUT_PATH = "./output/try1_" + RANDOM_CODE
os.makedirs(OUTPUT_PATH)

BATCH_SIZE = 64
EPOCHS = 100
PATIENCE = 5

CLASS_NAMES = ["Healhty", "TB"]
NUM_CLASS = 2
HIDDEN_SIZE = 256
AUDIO_LENGTH = 0.5 #0.89
SAMPLE_RATE = 22050
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

####################
# Data Prep
####################
df = pd.read_csv(f"{DATA_PATH}/{METADATA}")
df = df[df['path_file'].notna()]
df['path_file'] = DATA_PATH + df['path_file']
df = df[df['disease_label'].isin([0, 1])]
df = df[df['path_file'].apply(os.path.isfile)]
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df_0 = df[df['disease_label'] == 0].sample(n=df['disease_label'].value_counts().sort_index().values.min())
df_1 = df[df['disease_label'] == 1].sample(n=df['disease_label'].value_counts().sort_index().values.min())
#df_2 = df[df['disease_label'] == 2].sample(n=df['disease_label'].value_counts().sort_index().values.min())
df = pd.concat([df_0, df_1], ignore_index=True, sort=False)
print(df['disease_label'].value_counts())

train_df, test_df =  train_test_split(df, test_size=0.2, stratify=df['disease_label'], random_state=42)
sampler, class_weights = df_sampler(train_df)
class_weights = class_weights.to(DEVICE)

dataset_train = AudioDataset(DATA_PATH, train_df, sample_rate=SAMPLE_RATE, desired_length=AUDIO_LENGTH)
dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, pin_memory=True, shuffle=False, sampler=sampler, collate_fn=collate_fn)

dataset_test = AudioDataset(DATA_PATH, test_df, sample_rate=SAMPLE_RATE, desired_length=AUDIO_LENGTH)
dataloader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

DATAINPUT_SHAPE = next(iter(dataloader_train))[0].shape
print(DATAINPUT_SHAPE)

model = models.LSTMAudioClassifier1(input_size=DATAINPUT_SHAPE[2], seq_len=DATAINPUT_SHAPE[1], 
                            hidden_size=HIDDEN_SIZE, num_classes=NUM_CLASS).to(DEVICE)
#model = models.ResNet18(num_classes=NUM_CLASS).to(DEVICE)

criterion = nn.CrossEntropyLoss()#weight=class_weights)
optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

####################
# Train
####################
train_losses = []
val_losses = []

best_val_loss = float("inf")
best_model_path = f"{OUTPUT_PATH}/best_chk.pth"
epochs_no_improve = 0

for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0

    for feature_padded, labels, lengths in tqdm(dataloader_train, desc=f"Epoch {epoch+1}/{EPOCHS} - Training", leave=False):
        feature_padded = feature_padded.to(DEVICE)
        labels = labels.to(DEVICE)
        lengths = lengths.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(feature_padded)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(dataloader_train)
    train_losses.append(avg_train_loss)

    model.eval()
    total_val_loss = 0

    with torch.no_grad():
        for feature_padded, labels, lengths in tqdm(dataloader_test, desc=f"Epoch {epoch+1}/{EPOCHS} - Validation", leave=False):
            feature_padded = feature_padded.to(DEVICE)
            labels = labels.to(DEVICE)
            lengths = lengths.to(DEVICE)

            outputs = model(feature_padded)
            loss = criterion(outputs, labels)
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(dataloader_test)
    val_losses.append(avg_val_loss)
    scheduler.step(avg_val_loss)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"- Best model saved at epoch {epoch+1} with Val Loss: {best_val_loss:.4f}")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print("Early stopping triggered.")
            break

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(train_losses) + 1), train_losses, label="Training Loss", marker="o")
plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss", marker="s")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid()
plt.savefig(f"{OUTPUT_PATH}/loss_hist.png")

####################
# Testing
####################
model.load_state_dict(torch.load(best_model_path, weights_only=True, map_location='cpu'))
model.eval() 
all_preds, all_labels, all_probs  = [], [], []

with torch.no_grad():
    for feature_padded, labels, lengths in tqdm(dataloader_test, desc=f"Testing..", leave=False):
        feature_padded = feature_padded.to(DEVICE)
        labels = labels.to(DEVICE)
        lengths = lengths.to(DEVICE)
        
        outputs = model(feature_padded)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds) * 100
print(f"Validation Accuracy: {accuracy:.2f}%")

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig(f"{OUTPUT_PATH}/cm.png")

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
plt.savefig(f"{OUTPUT_PATH}/roc.png")

accuracy = accuracy_score(all_labels, all_preds, normalize=True)
b_accuracy = balanced_accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average="weighted")
try:
    roc_auc = roc_auc_score(all_labels, all_preds)
except Exception as exception:
    roc_auc = None

with open(f"{OUTPUT_PATH}/result.txt", "w") as file:
    file.write(df['disease_label'].value_counts().to_string() + "\n")
    file.write("\n")
    file.write(f"Accuracy {accuracy:.2f} | Balanced Accuracy {b_accuracy:.2f} | AUC {auc_score:.2f} |ROC AUC {roc_auc:.2f} | F1 Score Accuracy: {f1:.2f}\n")
 
print(f"Saved In: {OUTPUT_PATH}, Accuracy: {accuracy:.2f}, AUC: {auc_score:.2f}")

