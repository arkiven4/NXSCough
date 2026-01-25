#!/usr/bin/env python3
# Thesis Run
#  mfcc melspectogram logmel gammmaspectogram spectogram chroma
# NOW -> nfft1024
# Next -> Change folder logs, and retrrain for nfft2048

# PHASE 1
python train.py --init --model_name resnet34_mfcc --pooling_model ResNet34ManualClassifier --feature_type mfcc --feature_dim 13 --config_path configs/general.json 
python train.py --init --model_name resnet34_melspectogram --pooling_model ResNet34ManualClassifier --feature_type melspectogram --feature_dim 80 --config_path configs/general.json 
python train.py --init --model_name resnet34_logmel --pooling_model ResNet34ManualClassifier --feature_type logmel --feature_dim 80 --config_path configs/general.json 
python train.py --init --model_name resnet34_gtgram --pooling_model ResNet34ManualClassifier --feature_type gammmaspectogram --feature_dim 80 --config_path configs/general.json 
python train.py --init --model_name resnet34_spectogram --pooling_model ResNet34ManualClassifier --feature_type spectogram --feature_dim 513 --batch_size 64 --config_path configs/general.json 
python train.py --init --model_name resnet34_chroma --pooling_model ResNet34ManualClassifier --feature_type chroma --feature_dim 12 --config_path configs/general.json 

python train.py --init --model_name bilstm_mfcc --pooling_model BiLSTMClassifier --feature_type mfcc --feature_dim 13 --config_path configs/general.json
python train.py --init --model_name bilstm_melspectogram --pooling_model BiLSTMClassifier --feature_type melspectogram --feature_dim 80 --config_path configs/general.json
python train.py --init --model_name bilstm_logmel --pooling_model BiLSTMClassifier --feature_type logmel --feature_dim 80 --config_path configs/general.json
python train.py --init --model_name bilstm_gtgram --pooling_model BiLSTMClassifier --feature_type gammmaspectogram --feature_dim 80 --config_path configs/general.json
python train.py --init --model_name bilstm_spectogram --pooling_model BiLSTMClassifier --feature_type spectogram --feature_dim 513 --batch_size 32 --config_path configs/general.json
python train.py --init --model_name bilstm_chroma --pooling_model BiLSTMClassifier --feature_type chroma --feature_dim 12 --config_path configs/general.json

# logs_bilstmsattasp
python train.py --init --model_name bilstmatt_mfcc --pooling_model BiLSTMSelfAttASPClassifier --feature_type mfcc --feature_dim 13 --config_path configs/general.json
python train.py --init --model_name bilstmatt_melspectogram --pooling_model BiLSTMSelfAttASPClassifier --feature_type melspectogram --feature_dim 80 --config_path configs/general.json
python train.py --init --model_name bilstmatt_logmel --pooling_model BiLSTMSelfAttASPClassifier --feature_type logmel --feature_dim 80 --config_path configs/general.json
python train.py --init --model_name bilstmatt_gtgram --pooling_model BiLSTMSelfAttASPClassifier --feature_type gammmaspectogram --feature_dim 80 --config_path configs/general.json
python train.py --init --model_name bilstmatt_spectogram --pooling_model BiLSTMSelfAttASPClassifier --feature_type spectogram --feature_dim 1025 --batch_size 32 --config_path configs/general.json

# SSL
python train.py --init --model_name ssl_wavlm_peft --pooling_model PEFTWavLM_Try1 --feature_dim 1024 --config_path configs/general.json 
python train.py --init --model_name ssl_qwen_peft --pooling_model PEFTQwen3_Try1 --feature_dim 1024 --config_path configs/general.json 

python train.py --init --model_name bilstmselftattasp_gtgram --pooling_model BiLSTMSelfAttASPClassifier --feature_type gammmaspectogram --feature_dim 80 --config_path configs/general.json


# Main


python train.py --init --model_name resnet34_mfcc_delta --pooling_model ResNet34ManualClassifier --feature_type mfcc --feature_dim 26 --delta_feature --config_path configs/general.json 
python train.py --init --model_name resnet34_melspectogram_delta --pooling_model ResNet34ManualClassifier --feature_type melspectogram --feature_dim 160 --delta_feature --config_path configs/general.json 
python train.py --init --model_name resnet34_gtgram_delta --pooling_model ResNet34ManualClassifier --feature_type gammmaspectogram --feature_dim 160 --delta_feature --config_path configs/general.json 
python train.py --init --model_name resnet34_spectogram_delta --pooling_model ResNet34ManualClassifier --feature_type spectogram --feature_dim 1026 --delta_feature --config_path configs/general.json 

python train.py --init --model_name resnet34_mfcc_deltadelta --pooling_model ResNet34ManualClassifier --feature_type mfcc --feature_dim 39 --delta_feature --deltadelta_feature --config_path configs/general.json 
python train.py --init --model_name resnet34_melspectogram_deltadelta --pooling_model ResNet34ManualClassifier --feature_type melspectogram --feature_dim 240 --delta_feature --deltadelta_feature --config_path configs/general.json 
python train.py --init --model_name resnet34_gtgram_deltadelta --pooling_model ResNet34ManualClassifier --feature_type gammmaspectogram --feature_dim 240 --delta_feature --deltadelta_feature --config_path configs/general.json 
python train.py --init --model_name resnet34_spectogram_deltadelta --pooling_model ResNet34ManualClassifier --feature_type spectogram --feature_dim 1539 --delta_feature --deltadelta_feature --config_path configs/general.json 


python train.py --init --model_name bilstm_mfcc_delta --pooling_model BiLSTMClassifier --feature_type mfcc --feature_dim 26 --delta_feature --config_path configs/general.json
python train.py --init --model_name bilstm_melspectogram_delta --pooling_model BiLSTMClassifier --feature_type melspectogram --feature_dim 160 --delta_feature --config_path configs/general.json
python train.py --init --model_name bilstm_gtgram_delta --pooling_model BiLSTMClassifier --feature_type gammmaspectogram --feature_dim 160 --delta_feature --config_path configs/general.json
python train.py --init --model_name bilstm_spectogram_delta --pooling_model BiLSTMClassifier --feature_type spectogram --feature_dim 1026 --delta_feature --config_path configs/general.json

python train.py --init --model_name bilstm_mfcc_deltadelta --pooling_model BiLSTMClassifier --feature_type mfcc --feature_dim 39 --delta_feature --deltadelta_feature --config_path configs/general.json
python train.py --init --model_name bilstm_melspectogram_deltadelta --pooling_model BiLSTMClassifier --feature_type melspectogram --feature_dim 240 --delta_feature --deltadelta_feature --config_path configs/general.json
python train.py --init --model_name bilstm_gtgram_deltadelta --pooling_model BiLSTMClassifier --feature_type gammmaspectogram --feature_dim 240 --delta_feature --deltadelta_feature --config_path configs/general.json
python train.py --init --model_name bilstm_spectogram_deltadelta --pooling_model BiLSTMClassifier --feature_type spectogram --feature_dim 1539 --delta_feature --deltadelta_feature --config_path configs/general.json

# Extra
python train.py --init --model_name resnet34noaugment_gtgram --pooling_model ResNet34ManualClassifier --feature_type gammmaspectogram --feature_dim 80 --config_path configs/general.json 
python train.py --init --model_name resnet34noaugment_gtgram_delta --pooling_model ResNet34ManualClassifier --feature_type gammmaspectogram --feature_dim 160 --delta_feature --config_path configs/general.json 
python train.py --init --model_name resnet34noaugment_gtgram_deltadelta --pooling_model ResNet34ManualClassifier --feature_type gammmaspectogram --feature_dim 240 --delta_feature --deltadelta_feature --config_path configs/general.json 

python train.py --init --model_name resnet34multimask_gtgram --pooling_model ResNet34ManualClassifier --feature_type gammmaspectogram --feature_dim 80 --config_path configs/general.json 
python train.py --init --model_name resnet34multimask_gtgram_delta --pooling_model ResNet34ManualClassifier --feature_type gammmaspectogram --feature_dim 160 --delta_feature --config_path configs/general.json 
python train.py --init --model_name resnet34multimask_gtgram_deltadelta --pooling_model ResNet34ManualClassifier --feature_type gammmaspectogram --feature_dim 240 --delta_feature --deltadelta_feature --config_path configs/general.json 

############## New

python train.py --init --model_name resnet34_gtgram --pooling_model ResNet34ManualClassifier --feature_type gammmaspectogram --feature_dim 80 --config_path configs/general.json 
python train.py --init --model_name resnet34_gtgram_delta --pooling_model ResNet34ManualClassifier --feature_type gammmaspectogram --feature_dim 160 --delta_feature --config_path configs/general.json 
python train.py --init --model_name resnet34_gtgram_deltadelta --pooling_model ResNet34ManualClassifier --feature_type gammmaspectogram --feature_dim 240 --delta_feature --deltadelta_feature --config_path configs/general.json 

python train.py --init --model_name Res2Net_gtgram --pooling_model Res2NetVanilla --feature_type gammmaspectogram --feature_dim 80 --config_path configs/general.json 
python train.py --init --model_name Res2Net_gtgram_delta --pooling_model Res2NetVanilla --feature_type gammmaspectogram --feature_dim 160 --delta_feature --config_path configs/general.json 
python train.py --init --model_name Res2Net_gtgram_deltadelta --pooling_model Res2NetVanilla --feature_type gammmaspectogram --feature_dim 240 --delta_feature --deltadelta_feature --config_path configs/general.json 

python train.py --init --model_name resnet34branch_gtgram --pooling_model ResNet34SplitdeltaClassifier --feature_type gammmaspectogram --feature_dim 80 --config_path configs/general.json 
python train.py --init --model_name resnet34branch_gtgram_delta --pooling_model ResNet34SplitdeltaClassifier --feature_type gammmaspectogram --feature_dim 80 --delta_feature --config_path configs/general.json 
python train.py --init --model_name resnet34branch_gtgram_deltadelta --pooling_model ResNet34SplitdeltaClassifier --feature_type gammmaspectogram --feature_dim 80 --delta_feature --deltadelta_feature --config_path configs/general.json 


python train.py --init --model_name resnet34perfeaturenorm_gtgram_deltadelta --pooling_model ResNet34ManualClassifier --feature_type gammmaspectogram --feature_dim 240 --delta_feature --deltadelta_feature --config_path configs/general.json

python train.py --init --model_name resnet34perfeturenormbranch_gtgram_deltadelta --pooling_model ResNet34MultiEncoderClassifier --feature_type gammmaspectogram --feature_dim 80 --delta_feature --deltadelta_feature --config_path configs/general.json 


# 
python train.py --init --model_name bilstmattft_gtgram_deltadelta --pooling_model BiLSTMClassifier --feature_type gammmaspectogram --feature_dim 80 --config_path configs/general.json 

python train.py --init --model_name bilstmatt_gtgram --pooling_model BiLSTMClassifier --feature_type gammmaspectogram --feature_dim 80 --config_path configs/general.json



python train.py --init --model_name wavlm_peft --pooling_model PEFTQwen3_Try1 --config_path configs/general.json 


python train.py --init --model_name vit_gtgram --config_path configs/general.json 


python train.py --init --model_name SharedRes2NetTripleASP_gtgram_deltadelta --pooling_model SharedRes2NetTripleASP --feature_type gammmaspectogram --feature_dim 240 --delta_feature --deltadelta_feature --config_path configs/general.json 

python train.py --init --model_name resnet34Jitter_gtgram_deltadelta --pooling_model ResNet34ManualClassifier --feature_type gammmaspectogram --feature_dim 240 --delta_feature --deltadelta_feature --config_path configs/general.json 


python train.py --init --model_name resnet34perband_gtgram_deltadelta --pooling_model ResNet34ManualClassifier --feature_type gammmaspectogram --feature_dim 240 --delta_feature --deltadelta_feature --config_path configs/general.json 
####################################################################################

python zfinetune_ssl.py \
    --use_lora \
    --lora_r 32 \
    --train_csv /run/media/fourier/Data1/Pras/Database_ThesisNew/TB/metadata_cut_processed.csv.train \
    --test_csv /run/media/fourier/Data1/Pras/Database_ThesisNew/TB/metadata_cut_processed.csv.test \
    --audio_column file_path \
    --target_column disease_label \
    --output_dir logs/valence_model \
    --num_labels 2


python zfinetune_ssl2.py --init --model_name try_wavlmfinetune_full --config_path configs/ssl_finetuning.json
python ztrain_newgen.py --init --model_name baltrainset_effnet_mfccdd --config_path configs/lstm_cnn.json


0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0
for f in 0.02 0.05 0.1 0.2 0.4 0.6 0.8 1.0; do
  python ztrain_newgen.py \
    --init \
    --model_name apsipa_lstm_frac${f} \
    --config_path configs/lstm_cnn.json \
    --fraction ${f}
done

  python ztrain_newgen.py \
    --init \
    --model_name apsipa_lstm_frac${f} \
    --config_path configs/lstm_cnn.json \
    --fraction 1.0


python train.py --init --model_name combined3db_augment_qwenencoder --config_path configs/general.json
python train_coughdetect.py --init --model_name ssl_cough --config_path configs/cough_detection.json

python evaluate.py --model_name nonstatify_disen_resnet

for d in logs/*nonstatify*/; do
    model_name=$(basename "$d")
    python evaluate.py --model_name "$model_name"
done