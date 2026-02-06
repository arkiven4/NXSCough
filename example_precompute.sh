#!/bin/bash
# Example script for precomputing and using features

# Configuration
CONFIG_PATH="configs/ssl_finetuning.json"
MODEL_NAME="experiment_with_precomputed"
PRECOMPUTED_DIR="./precomputed_features/melspec"

echo "=== Step 1: Initial Training (without precomputed) ==="
echo "This will compute features on-the-fly during training"
python train_fastrecov.py \
  --init \
  --config_path "$CONFIG_PATH" \
  --model_name "${MODEL_NAME}_initial" \
  --feature_type melspectogram

echo ""
echo "=== Step 2: Precompute Features ==="
echo "This will extract and save all features to disk"
echo "wav_stats.pickle will be automatically computed from training data"
python precompute_features.py \
  --config "logs/${MODEL_NAME}_initial/config.json" \
  --output_dir "$PRECOMPUTED_DIR"

echo ""
echo "=== Step 3: Training with Precomputed Features ==="
echo "This will use precomputed features (MUCH FASTER!)"
python train_fastrecov.py \
  --config_path "logs/${MODEL_NAME}_initial/config.json" \
  --model_name "${MODEL_NAME}_fast" \
  --use_precomputed \
  --precomputed_dir "$PRECOMPUTED_DIR"

echo ""
echo "=== Done! ==="
echo "Compare training times:"
echo "  Initial: see logs/${MODEL_NAME}_initial/"
echo "  With precomputed: see logs/${MODEL_NAME}_fast/"
