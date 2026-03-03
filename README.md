# 🔊 NXSCough: NEXUS Cough Classification

End-to-end training pipeline for cough-based disease classification using spectro-temporal features, deep models (BiLSTM, ResNet), and optional active-learning style filtering (FastReCoV).

---

## ✨ Highlights

- Multiple feature frontends: `mfcc`, `melspectogram`, `logmel`, `gammmaspectogram`, `spectogram`, `fbank_ast`
- Multiple model backbones: `BiLSTMSelfAttASPClassifier`, `ResNet34ManualClassifier`, SSL/PEFT variants
- Precompute + cached feature training support
- Hyperparameter search with Optuna
- K-Fold-ready training/evaluation workflow
- Optional FastReCoV sample filtering
- TensorBoard-compatible logs

---

## 📁 Repository Layout

- `train.py`: main training/evaluation script
- `train_hypersearch.py`: Optuna-based hyperparameter search
- `precompute_features.py`: offline feature extraction and caching
- `train_fastrecov.py`: FastReCoV candidate discovery
- `train_fastrecov_positive.py`: positive-focused FastReCoV variant
- `configs/general.json`: main experiment config
- `run.sh`: experiment command bank (precompute/search/train/ablation)
- `logs/`: model checkpoints, configs, outputs
- `precomputed_features/`: cached features for faster runs

---

## 🚀 Quick Start

### 1) Clone and install

```bash
git clone https://github.com/arkiven4/NXSCough
cd NXSCough
pip install -r requirements.txt
```

### 2) Configure dataset path

Edit `configs/general.json`:

```json
"data": {
  "db_path": "/path/to/your/dataset/root",
  "metadata_csv": "metadata_combine.csv",
  "target_column": "disease_status"
}
```

Expected dataset root layout:

```text
/path/to/your/dataset/root/
├── metadata_combine.csv
└── CombineData/
    ├── sample1.wav
    ├── sample2.wav
    └── ...
```

> The metadata file should include at least `path_file`, label column (`disease_status` by default), and optionally participant/tabular columns referenced by `column_order` in `configs/general.json`.

### 3) Precompute features (recommended)

```bash
python precompute_features.py \
  --config configs/general.json \
  --output_dir ./precomputed_features/logmel \
  --feature_type logmel
```

### 4) Train baseline model

```bash
python train.py --init \
  --model_name bilstmbest_logmel \
  --pooling_model BiLSTMSelfAttASPClassifier \
  --feature_type logmel \
  --feature_dim 80 \
  --config_path configs/general.json \
  --use_precomputed \
  --precomputed_dir ./precomputed_features/logmel
```

### 5) Evaluate trained model

```bash
python train.py --eval --model_name bilstmbest_logmel
```

---

## 🧪 Common Experiment Recipes

### A) Hyperparameter Search

```bash
python train_hypersearch.py --init \
  --model_name searchlstm10fold_logmel \
  --pooling_model BiLSTMSelfAttASPClassifier \
  --feature_type logmel \
  --feature_dim 80 \
  --config_path configs/general.json \
  --use_precomputed \
  --precomputed_dir ./precomputed_features/logmel
```

### B) ResNet Baseline

```bash
python train.py --init \
  --model_name resnetbest_logmel \
  --pooling_model ResNet34ManualClassifier \
  --feature_type logmel \
  --feature_dim 80 \
  --config_path configs/general.json \
  --use_precomputed \
  --precomputed_dir ./precomputed_features/logmel
```

### C) FastReCoV Flow

1. Generate FastReCoV outputs:

```bash
python train_fastrecov.py --init \
  --model_name fastrecov5_wavsfolds_originalweight \
  --pooling_model BiLSTMSelfAttASPClassifier \
  --feature_type logmel \
  --feature_dim 80 \
  --config_path configs/general.json \
  --use_precomputed \
  --precomputed_dir ./precomputed_features/logmel
```

2. Train with FastReCoV filtering:

```bash
python train.py --init \
  --model_name bilstmrecov_logmel \
  --pooling_model BiLSTMSelfAttASPClassifier \
  --feature_type logmel \
  --feature_dim 80 \
  --config_path configs/general.json \
  --use_precomputed \
  --precomputed_dir ./precomputed_features/logmel \
  --use_fastrecov \
  --fastrecov_dir logs/fastrecov5_wavsfolds_originalweight
```

---

## 📊 Monitoring

```bash
tensorboard --logdir ./logs
```

Open the displayed local URL (usually `http://localhost:6006`) to track loss/metrics and compare runs.

---

## 🧠 SSL / PEFT Examples

WavLM PEFT:

```bash
python train.py --init \
  --model_name wavlmasp_peft \
  --pooling_model PEFTWavLM_Try1 \
  --feature_dim 1024 \
  --config_path configs/general.json
```

Qwen PEFT:

```bash
python train.py --init \
  --model_name qwenasp_peft \
  --pooling_model PEFTQwen3_Try1 \
  --feature_dim 1024 \
  --config_path configs/general.json
```

---

## 🐳 Singularity (Optional)

Build image:

```bash
singularity build nxscough.sif nxscough.def
```

Run training with host dataset mounted to `/mnt/data`:

```bash
singularity exec --bind /host/dataset/root:/mnt/data --nv nxscough.sif \
  python3 train.py --init \
  --model_name bilstmbest_logmel \
  --pooling_model BiLSTMSelfAttASPClassifier \
  --feature_type logmel \
  --feature_dim 80 \
  --config_path configs/general.json
```

Inside the container, set `db_path` in `configs/general.json` to `/mnt/data`.

---

## ⚠️ Notes

- Use feature names exactly as implemented in this repo (including current spellings):
  - `melspectogram`, `gammmaspectogram`, `spectogram`
- `run.sh` is a command collection for many experiments; run only the blocks you need.
- Logs and checkpoints are saved under `logs/<model_name>/`.

---

## 🙏 Acknowledgement

Optional RIR augmentation can use the OpenSLR RIRS dataset:
https://www.openslr.org/28/
