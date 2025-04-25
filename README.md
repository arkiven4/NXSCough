# 🔊 NEXUS Cough Classification

This repository contains a training pipeline for a Cough classification model using Various Model. It supports audio augmentation via the [RIRS noise dataset](https://www.openslr.org/28/) for realistic room simulation.

---

## 🚀 Getting Started

### 1. Preparation
Install dependencies with:
```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python ztrain_nonssl.py
```

Training logs will be saved to the ./logs/ directory. You can Check with this command:


```sh
tensorboard --logdir ./logs/lstm_try1
```
Open the provided [localhost](http://localhost:6006) link in your browser to monitor loss, accuracy, and other metrics during training.

After training is complete, please check the `logs` folder.

### Optional: Data Augmentation with RIRS

To apply room impulse response (RIR) augmentation:

Download the RIRS dataset from:
👉 https://www.openslr.org/28/

Extract the contents and place them in a suitable path used by your config or augmentation script.
