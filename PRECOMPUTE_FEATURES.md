# Precomputed Features Guide

This guide explains how to use precomputed features to speed up training by calculating acoustic features once and reusing them across multiple training runs.

## Overview

Instead of computing acoustic features (MFCC, mel-spectrogram, etc.) during each training run, you can:
1. **Precompute** features once and save them to disk
2. **Load** precomputed features directly during training

This significantly speeds up training, especially for:
- Multiple training runs with same features
- Hyperparameter tuning
- Cross-validation experiments

## Benefits

- **Faster training**: Skip feature computation during each epoch
- **Consistency**: Same features used across all experiments
- **Efficiency**: Compute once, use many times
- **Flexibility**: Easy to experiment with different models on same features

## Usage

### Step 1: Precompute Features

First, compute and save features for your dataset:

```bash
python precompute_features.py \
  --config configs/your_config.json \
  --output_dir ./precomputed_features
```

**Arguments:**
- `--config`: Path to your training configuration JSON file
- `--output_dir`: Directory where features will be saved
- `--train_csv`: (Optional) Override training CSV path
- `--test_csv`: (Optional) Override test CSV path

**Output:**
The script will create:
- `wav_stats.pickle`: Normalization statistics computed from training data
- `feature_mapping_train.csv`: Maps audio files to feature file paths
- `feature_mapping_test.csv`: Test set mapping (if available)
- `feature_metadata_train.json`: Feature extraction parameters
- `*.pt` files: PyTorch tensors containing precomputed features

### Step 2: Train with Precomputed Features

Use the precomputed features during training:

```bash
python train_fastrecov.py \
  --config_path configs/your_config.json \
  --model_name my_experiment \
  --use_precomputed \
  --precomputed_dir ./precomputed_features
```

**New Arguments:**
- `--use_precomputed`: Enable loading of precomputed features
- `--precomputed_dir`: Directory containing precomputed features

## Example Workflow

```bash
# 1. First training run: compute features
python train_fastrecov.py \
  --init \
  --config_path configs/ssl_finetuning.json \
  --model_name experiment_01 \
  --feature_type melspectogram

# 2. After training, precompute features for reuse
python precompute_features.py \
  --config logs/experiment_01/config.json \
  --output_dir ./features/melspec_features

# 3. Subsequent runs use precomputed features (much faster!)
python train_fastrecov.py \
  --config_path logs/experiment_01/config.json \
  --model_name experiment_02 \
  --use_precomputed \
  --precomputed_dir ./features/melspec_features
```

## Important Notes

### Feature Consistency
- **Must match**: Precomputed features must match your config settings:
  - `feature_type` (mfcc, melspectogram, chroma, etc.)
  - `sampling_rate`
  - `n_mel_channels`
  - `hop_length`, `win_length`, `filter_length`
  - `delta_feature`, `deltadelta_feature`
  - Normalization settings

### Limitations
- **No augmentation**: Audio augmentation (speed, pitch, noise) is skipped with precomputed features
- **Fixed features**: Cannot change feature extraction parameters without recomputing
- **Storage**: Requires disk space for all feature files

### When to Use
✅ **Use precomputed features when:**
- Training multiple models with same features
- Doing hyperparameter tuning on model architecture
- Running repeated experiments for statistical analysis
- Working with expensive feature extraction

❌ **Don't use precomputed features when:**
- Experimenting with different feature types
- Using heavy data augmentation
- Need to change feature extraction parameters
- Initial exploration phase

## Troubleshooting

### Error: "feature_path column not found"
- Make sure `feature_mapping_train.csv` has a `path_file` column matching your dataset
- Check that the CSV was generated with the correct dataset

### Error: "Feature file not found"
- Verify that `--precomputed_dir` points to the correct directory
- Check that all `.pt` files exist in the precomputed directory

### Slower than expected
- Ensure features are on same device/filesystem as training
- Check that I/O is not bottlenecked (use SSD if possible)
- Verify features are actually being loaded (check logs)

## Advanced: Custom Feature Paths

If you need to use custom feature mappings:

```python
# In your training code
dataset = CoughDatasets(
    data.values,
    hparams,
    train=True,
    use_precomputed=True
)
# Set custom column index where feature paths are stored
dataset.set_feature_path_column(column_index)
```

## Performance Comparison

Typical speedup (depends on feature complexity):

| Feature Type | Without Precompute | With Precompute | Speedup |
|-------------|-------------------|-----------------|---------|
| MFCC        | ~0.8 sec/batch    | ~0.2 sec/batch  | 4x      |
| Mel-spec    | ~1.2 sec/batch    | ~0.2 sec/batch  | 6x      |
| OpenSMILE   | ~2.5 sec/batch    | ~0.2 sec/batch  | 12x     |

*Results on example system with 28 workers, batch size 32*
