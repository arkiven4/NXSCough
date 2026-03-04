#!/usr/bin/env python3
"""
Precompute acoustic features for cough audio dataset.
This script processes audio files and saves features to disk for faster training.

This script uses CoughDatasets class to ensure feature extraction consistency
between precomputation and training.
"""

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import utils
from cough_datasets import CoughDatasets


def precompute_features(df, hparams, output_dir, split_name="train"):
    """Precompute features for all audio files in the dataframe using CoughDatasets."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    wav_stats_path = None
    if split_name == "train" and hparams.mean_std_norm:
        wav_stats_path = output_dir / "wav_stats.pickle"
        print(f"Computing wav statistics for normalization...")
        utils.compute_spectrogram_stats_from_dataset(
            df,
            hparams,
            pickle_path=str(wav_stats_path)
        )
        print(f"Saved wav stats to {wav_stats_path}")
    elif split_name != "train" and hparams.mean_std_norm:
        wav_stats_path = output_dir / "wav_stats.pickle"
        if wav_stats_path.exists():
            print(f"Using existing wav stats from {wav_stats_path}")
        else:
            print(f"Warning: wav_stats not found at {wav_stats_path}, proceeding without normalization")
            wav_stats_path = None
    
    print(f"Initializing CoughDatasets for feature extraction ({split_name} split)...")
    dataset = CoughDatasets(
        df.values,
        hparams,
        train=False,  # Disable augmentation during precomputation
        wav_stats_path=str(wav_stats_path) if wav_stats_path else None,
        use_precomputed=False  # We're computing features, not loading them
    )
    
    feature_paths = []
    print(f"Precomputing features for {len(df)} files ({split_name} split)...")
    
    for idx in tqdm(range(len(dataset)), total=len(dataset)):
        try:
            # Get the audio path
            row = df.iloc[idx]
            if hparams.cough_detection:
                wavname = row[0]
            else:
                wavname = row[0]
            
            audio_path = os.path.join(hparams.db_path, wavname)
            
            # Use CoughDatasets.get_audio() to extract features
            # This ensures consistency with training
            dse_id = row[1] if len(row) > 1 else None
            features, _ = dataset.get_audio(audio_path, dse_id=dse_id)
            
            # Remove batch dimension if present
            if features.ndim == 4:  # [B, C, H, W]
                features = features.squeeze(0)
            elif features.ndim == 3:  # [B, C, T]
                features = features.squeeze(0)
            
            # Create output path
            # Use relative path structure to avoid collisions
            rel_path = wavname.replace('/', '_').replace('\\', '_')
            feature_filename = f"{idx}_{Path(rel_path).stem}.pt"
            feature_path = output_dir / feature_filename
            
            # Save features
            torch.save(features, feature_path)
            feature_paths.append(str(feature_path))
        except Exception as e:
            print(f"Error processing {wavname}: {e}")
            import traceback
            traceback.print_exc()
            feature_paths.append(None)
    
    # Create mapping dataframe
    df_mapping = df.copy()
    df_mapping['feature_path'] = feature_paths
    
    # Save mapping
    mapping_path = output_dir / f"feature_mapping_{split_name}.csv"
    df_mapping.to_csv(mapping_path, index=False)
    print(f"Saved feature mapping to {mapping_path}")
    
    # Save metadata
    metadata = {
        'feature_type': hparams.feature_type,
        'sampling_rate': hparams.sampling_rate,
        'n_mel_channels': getattr(hparams, 'n_mel_channels', None),
        'hop_length': hparams.hop_length,
        'win_length': hparams.win_length,
        'filter_length': hparams.filter_length,
        'delta_feature': hparams.delta_feature,
        'deltadelta_feature': hparams.deltadelta_feature,
        'mean_std_norm': hparams.mean_std_norm,
        'per_band_norm': hparams.per_band_norm,
        'num_samples': len(df),
        'valid_samples': sum(1 for p in feature_paths if p is not None)
    }
    
    metadata_path = output_dir / f"feature_metadata_{split_name}.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")
    
    return feature_paths


def main():
    parser = argparse.ArgumentParser(description="Precompute features for cough dataset")
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to config JSON file')
    parser.add_argument('-o', '--output_dir', type=str, required=True, help='Output directory for features')
    parser.add_argument('--feature_type', type=str, default=None, help='Override feature type (mfcc, melspectogram, logmel, etc.)')
    parser.add_argument('--train_csv', type=str, default=None, help='Override train CSV path')
    parser.add_argument('--test_csv', type=str, default=None, help='Override test CSV path')
    
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)

    if args.feature_type:
        config['data']['feature_type'] = args.feature_type
        print(f"Overriding feature_type to: {args.feature_type}")
    
    hparams = utils.HParams(**config)
    if args.train_csv:
        df_train = pd.read_csv(args.train_csv)
    else:
        df_train = pd.read_csv(f'data/{hparams.data.metadata_csv}.train')
    df_train = df_train.reset_index(drop=True)
    df_train = df_train[hparams.data.column_order]
    
    print(f"Loaded {len(df_train)} training samples")
    precompute_features(
        df_train, hparams.data,
        args.output_dir, split_name="train")
    
    try:
        if args.test_csv:
            df_test = pd.read_csv(args.test_csv)
        else:
            df_test = pd.read_csv(f'/run/media/fourier/Data1/Pras/Thesis_Nexus/NXSCough/data/{hparams.data.metadata_csv}.test')
        df_test = df_test.reset_index(drop=True)
        
        print(f"\nLoaded {len(df_test)} test samples")
        precompute_features(
            df_test,
            hparams.data,
            args.output_dir,
            split_name="test"
        )
    except FileNotFoundError:
        print("\nNo test set found, skipping...")
    
    print("\n✓ Feature precomputation complete!")
    print(f"Features saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
