import os, pickle, librosa, pickle

import numpy as np
import pandas as pd
from tqdm import tqdm
import opensmile
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyClassifier

import utils

DATA_PATH = "/run/media/fourier/Data1/Pras/Database_ThesisNew/"
METADATA = "tb_solic_filt.csv" # tb_all_filt.csv
RANDOM_CODE = utils.generate_random_code()
OUTPUT_PATH = "./output/try2_" + RANDOM_CODE
os.makedirs(OUTPUT_PATH)

CLASS_NAMES = ["Healhty", "TB"]
NUM_CLASS = 2
HIDDEN_SIZE = 256
AUDIO_LENGTH = 0.5
SAMPLE_RATE = 22050

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.Functionals,
)

####################
# Data Prep
####################
with open(f"{DATA_PATH}/norm_stat.pkl", 'rb') as f:
    wav_mean, wav_std = pickle.load(f)
    print("Loaded Norm Stats")

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

X_features = []
y_label = []

for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    signal, sampling_rate = librosa.load(row['path_file'], sr=SAMPLE_RATE)
    signal = signal / 32768.0
    signal = (signal - wav_mean) / (wav_std + 0.000001)

    smile_feature = smile.process_signal(signal, sampling_rate).values.reshape(-1)
    if np.isnan(np.sum(smile_feature)) == False:
        X_features.append(smile.process_signal(signal, sampling_rate).values.reshape(-1))
        y_label.append(row['disease_label'])

X_features = np.stack(X_features)
y_label = np.stack(y_label)

scaler = StandardScaler()
scaler.fit(X_features)
X_features = scaler.transform(X_features)

X_train, X_test, y_train, y_test = train_test_split(X_features, y_label, stratify=y_label, random_state=42)

print("Training Lazy Predict.......")
clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)

with open(f"{OUTPUT_PATH}/saved_models.pickle", 'wb') as handle:
    pickle.dump(clf, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(f"{OUTPUT_PATH}/result.txt", "w") as file:
    file.write(df['disease_label'].value_counts().to_string() + "\n")
    file.write("\n")
    file.write(models.to_string())

print(models)
print(f"Saved In: {OUTPUT_PATH}")
# clf.models['AdaBoostClassifier'].predict(X_test)