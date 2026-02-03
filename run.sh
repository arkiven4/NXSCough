#!/usr/bin/env python3
# Thesis Run
#  mfcc melspectogram logmel gammmaspectogram spectogram chroma fbank_ast
# NOW -> nfft1024
# Next -> Change folder logs, and retrrain for nfft2048

############################################################### PHASE 1 ###############################################################
python train.py --init --model_name resnet34_mfcc --pooling_model ResNet34ManualClassifier --feature_type mfcc --feature_dim 13 --config_path configs/general.json 
python train.py --init --model_name resnet34_melspectogram --pooling_model ResNet34ManualClassifier --feature_type melspectogram --feature_dim 80 --config_path configs/general.json 
python train.py --init --model_name resnet34_logmel --pooling_model ResNet34ManualClassifier --feature_type logmel --feature_dim 80 --config_path configs/general.json 
python train.py --init --model_name resnet34_gtgram --pooling_model ResNet34ManualClassifier --feature_type gammmaspectogram --feature_dim 80 --config_path configs/general.json 
python train.py --init --model_name resnet34_spectogram --pooling_model ResNet34ManualClassifier --feature_type spectogram --feature_dim 1025 --batch_size 64 --config_path configs/general.json 

python train.py --init --model_name bilstm_mfcc --pooling_model BiLSTMSelfAttASPClassifier --feature_type mfcc --feature_dim 13 --config_path configs/general.json
python train.py --init --model_name bilstm_melspectogram --pooling_model BiLSTMSelfAttASPClassifier --feature_type melspectogram --feature_dim 80 --config_path configs/general.json
python train.py --init --model_name bilstm_logmel --pooling_model BiLSTMSelfAttASPClassifier --feature_type logmel --feature_dim 80 --config_path configs/general.json
python train.py --init --model_name bilstm_gtgram --pooling_model BiLSTMSelfAttASPClassifier --feature_type gammmaspectogram --feature_dim 80 --config_path configs/general.json
python train.py --init --model_name bilstm_spectogram --pooling_model BiLSTMSelfAttASPClassifier --feature_type spectogram --feature_dim 1025 --batch_size 32 --config_path configs/general.json
# python train.py --init --model_name bilstm_spectogram_deltadelta --pooling_model BiLSTMClassifier --feature_type spectogram --feature_dim 1539 --delta_feature --deltadelta_feature --config_path configs/general.json

python train.py --init --model_name resnet34re_logmel --pooling_model ResNet34ManualClassifier --feature_type logmel --feature_dim 80 --config_path configs/general.json 
############################################################### SSL ###############################################################
# "lora_rank": 64, "lora_alpha": 128,
python train.py --init --model_name wavlmasp_peft --pooling_model PEFTWavLM_Try1 --feature_dim 1024 --config_path configs/general.json 

# "lora_rank": 16, "lora_alpha": 32,
python train.py --init --model_name qwenasp_peft --pooling_model PEFTQwen3_Try1 --feature_dim 1024 --config_path configs/general.json 

python train.py --init --model_name ast_try1 --pooling_model AST_Try1 --feature_type fbank_ast --feature_dim 120 --config_path configs/general.json 

###################################################################################################################################
python train.py --init --model_name resnet34nosampler_logmel --pooling_model ResNet34ManualClassifier --feature_type logmel --feature_dim 80 --config_path configs/general.json 
python train.py --init --model_name resnet34pal_logmel --pooling_model ResNet34ManualClassifier --feature_type logmel --feature_dim 80 --config_path configs/general.json 
# TODO: mutimask effect, no augment effect


python train.py --init --model_name resnet34prl_logmel --pooling_model ResNet34ManualClassifier --feature_type logmel --feature_dim 80 --config_path configs/general.json 
###################################################################################################################################
# TODO: MIXUP,  

