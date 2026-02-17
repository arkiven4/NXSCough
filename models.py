import torch
import math
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from efficientnet_pytorch import EfficientNet
from torch.nn.utils.rnn import pad_sequence
from torchcrf import CRF
import numpy as np
import gc
import torch


import torchvision
from transformers import AutoConfig, AutoFeatureExtractor

from typing import Any, Callable, Optional
from collections import OrderedDict

from functools import partial

from timm.models.swin_transformer import SwinTransformerBlock
from timm.models.vision_transformer import Block

# from s3prl.upstream.mockingjay.builder import PretrainedTransformer

import modules
import commons
import layers

# TODO : AttentiveStatisticsPooling
# class Baseline(nn.Module):
#     """
#     AttentiveStatisticsPooling
#     Paper: Attentive Statistics Pooling for Deep Speaker Embedding
#     Link: https://arxiv.org/pdf/1803.10963.pdf
#     """

#     def __init__(self, input_size, **kwargs):
#         super().__init__()
#         self._indim = input_size
#         self.sap_linear = nn.Linear(input_size, input_size)
#         self.attention = nn.Parameter(torch.FloatTensor(input_size, 1))
#         torch.nn.init.normal_(self.attention, mean=0, std=1)

#     def compute_length_from_mask(self, mask):
#         """
#         mask: (batch_size, T)
#         Assuming that the sampling rate is 16kHz, the frame shift is 20ms
#         """
#         wav_lens = torch.sum(mask, dim=1)  # (batch_size, )
#         feat_lens = torch.div(wav_lens-1, 16000*0.02, rounding_mode="floor") + 1
#         feat_lens = feat_lens.int().tolist()
#         return feat_lens

#     def forward(self, xs, attention_mask, **kwargs):
#         """
#         xs: (batch_size, T, feat_dim)
#         mask: (batch_size, T)

#         => output: (batch_size, feat_dim*2)
#         """
#         feat_lens = self.compute_length_from_mask(attention_mask)
#         pooled_list = []
#         for x, feat_len in zip(xs, feat_lens):
#             x = x[:feat_len].unsqueeze(0)
#             h = torch.tanh(self.sap_linear(x))
#             w = torch.matmul(h, self.attention).squeeze(dim=2)
#             w = F.softmax(w, dim=1).view(x.size(0), x.size(1), 1)
#             mu = torch.sum(x * w, dim=1)
#             rh = torch.sqrt(
#                 (torch.sum((x**2) * w, dim=1) - mu**2).clamp(min=1e-5))
#             x = torch.cat((mu, rh), 1).squeeze(0)
#             pooled_list.append(x)

#         # nn.Dropout(p=0.1),
#         # Maxout(d_in = config.embeddings, d_out = config.hidden_size, pool_size = 3),
#         return torch.stack(pooled_list)

class ResNet34ManualClassifier(nn.Module):
    def __init__(
        self,
        feature_dim: int = 39,
        resnet_type: str = "resnet18",  # ["resnet18", "resnet34", "resnet50", "resnet101"]
        num_layers_resnet: int = 4,
        hidden_dim_classifier: int = 128,
        dropout: float = 0.5,
        output_dim: int = 1, **kwargs
    ):
        super().__init__()

        self.encoder1 = modules.Resnet34Manual(resnet_type=resnet_type, feature_dim=feature_dim, num_layers=num_layers_resnet)
        self.pool_out_dim = self.encoder1.pool.get_out_dim()

        self.classifier = nn.Sequential(
            nn.Linear(self.pool_out_dim, hidden_dim_classifier),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_classifier, output_dim)
        )

    def forward(self, x, tabular_ids=None, **kwargs):
        """
        x: (B, n_mels, T) mel-spectrogram frames
        """
        x = x.unsqueeze_(1)
        stats = self.encoder1(x)  # torch.Size([128, 5120])
        disease_logits = self.classifier(stats)

        return {
            "disease_logits": disease_logits,
            "embeddings": stats,
        }

class ResNet34CBAMClassifier(nn.Module):
    def __init__(
        self,
        dummy_input,
        feature_dim: int = 39,
        output_dim: int = 2, **kwargs
    ):
        super().__init__()

        #model = ResNet(BasicBlock, [2, 2, 2, 2], network_type, num_classes, att_type)
        #model = ResNet(BasicBlock, [3, 4, 6, 3], network_type, num_classes, att_type)
        #model = ResNet(Bottleneck, [3, 4, 6, 3], network_type, num_classes, att_type)
        #model = ResNet(Bottleneck, [3, 4, 23, 3], network_type, num_classes, att_type)
        # ['BAM', 'CBAM']

        self.encoder1 = modules.ResNetCBAM(feature_dim, [3, 4, 6, 3], "Dummy", 'CBAM')

        self.classifier = nn.Sequential(
            nn.Linear(self.encoder1.afterpooling, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, output_dim)
        )

    def forward(self, x, tabular_ids=None, **kwargs):
        """
        x: (B, n_mels, T) mel-spectrogram frames
        """
        x = x.unsqueeze_(1)
        stats = self.encoder1(x)  # torch.Size([128, 5120])
        disease_logits = self.classifier(stats)

        return {
            "disease_logits": disease_logits,
            "embeddings": stats,
        }

class LSTMAudioClassifier1(nn.Module):
    def __init__(self, dummy_input, feature_dim = 80, hidden_size = 1024, output_dim = 1, **kwargs):
        super(LSTMAudioClassifier1, self).__init__()
        
        self.batch_norm1 = nn.BatchNorm1d(feature_dim)
        self.lstm1 = nn.LSTM(feature_dim, hidden_size, batch_first=True)
        
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        
        #self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8, batch_first=True)

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, x, tabular_ids=None, **kwargs):
        """
        x: (B, n_mels, T) mel-spectrogram frames
        """
        x = x.permute(0, 2, 1)
        x = self.batch_norm1(x.transpose(1, 2)).transpose(1, 2)
        x, _ = self.lstm1(x)
        
        x = self.batch_norm2(x.transpose(1, 2)).transpose(1, 2)
        x, _ = self.lstm2(x)
        
        # attn_output, _ = self.attention(x, x, x)
        # x = torch.mean(attn_output, dim=1)

        x = self.flatten(x[:, -1, :])
        x = self.dropout(x)

        embedding = x.clone()
        disease_logits = self.fc(x)
        return {
            "disease_logits": disease_logits,
            "embeddings": embedding,
        }

class BiLSTMSelfAttASPClassifier(nn.Module):
    def __init__(
        self,
        feature_dim: int = 39,
        hidden_size: int = 512,
        lstmnum_layers: int = 2,
        att_head: int = 2,
        hidden_dim_classifier: int = 128,
        dropout: float = 0.1,
        output_dim: int = 1, 
        use_tabular: bool = False,
        fusion_type: str = "cross_attn",  # ["gating", "cross_attn", "film"]
        att_head_fusion: int = 2,
        **kwargs
    ):
        super().__init__()
        
        self.use_tabular = use_tabular
        self.fusion_type = fusion_type
        fusion_dim = hidden_size * 4

        # ========== AUDIO BACKBONE ==========
        # self.frontend = modules.FeatureFrontend(in_freq=feature_dim, out_dim=int(feature_dim * 1.6))
        # feature_dim = int(feature_dim * 1.6)

        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=lstmnum_layers,
            dropout=0.2 if lstmnum_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )

        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=att_head,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_size * 2)
        self.pool = modules.AttentiveStatisticsPooling(channels=hidden_size * 2)

        # ========== TABULAR PROJECTION ==========
        if self.use_tabular:
            print("------------- USE TABULAR --------------------")
            self.tab_proj = nn.Sequential(
                nn.Linear(4, fusion_dim),
                nn.ReLU(),
                nn.LayerNorm(fusion_dim)
            )

        # ========== FUSION MECHANISMS ==========
        if self.use_tabular and fusion_type == "gating":
            self.gate = nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim),
                nn.Sigmoid()
            )

        if self.use_tabular and fusion_type == "film":
            self.film = nn.Linear(fusion_dim, fusion_dim * 2)

        if self.use_tabular and fusion_type == "cross_attn":
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=fusion_dim,
                num_heads=att_head_fusion,
                batch_first=True
            )
            self.ca_norm = nn.LayerNorm(fusion_dim)

        # ========== CLASSIFIER ==========
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim_classifier),
            nn.BatchNorm1d(hidden_dim_classifier),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_classifier, output_dim)
        )

    def forward(self, x, tabular_ids=None, train=False, **kwargs):
        """
        x: (B, n_mels, T) mel-spectrogram frames
        tabular_ids: (B, tab_dim)
        """
        # ===== AUDIO ENCODING =====
        x = x.permute(0, 2, 1)
        #x, freq_weights = self.frontend(x)
        audio_feat, _ = self.lstm(x)           # (B, T, 2H)

        self_attn_out, self_attn_weights = self.attn(audio_feat, audio_feat, audio_feat, need_weights=True)
        audio_feat = self.norm(audio_feat + self_attn_out)
        
        audio_feat, asp_weights = self.pool(audio_feat.permute(0, 2, 1)) # torch.Size([128, 2048])
        fused = audio_feat

        # ===== HYBRID MODE =====
        if self.use_tabular and tabular_ids is not None:
            tab_feat = self.tab_proj(tabular_ids)

            # if self.training:
            #     r = torch.rand(1, device=audio_feat.device)
            #     # if r < 0.2:
            #     #     audio_feat = torch.zeros_like(audio_feat)
            #     if r < 0.2 + 0.2:
            #         tab_feat = torch.zeros_like(tab_feat)

            # ---- Dynamic Gating ----
            if self.fusion_type == "gating":
                alpha = self.gate(tab_feat)
                fused = alpha * audio_feat + (1 - alpha) * tab_feat

            # ---- FiLM ----
            elif self.fusion_type == "film":
                gamma, beta = self.film(tab_feat).chunk(2, dim=-1)
                fused = gamma * audio_feat + beta

            # ---- Cross-Attention ----
            elif self.fusion_type == "cross_attn":
                q = tab_feat.unsqueeze(1)     # (B, 1, D)
                k = audio_feat.unsqueeze(1)   # (B, 1, D)
                v = audio_feat.unsqueeze(1)

                ca_out, ca_weights = self.cross_attn(q, k, v)
                fused = self.ca_norm(audio_feat + ca_out.squeeze(1))

        disease_logits = self.classifier(fused)
        return {
            "disease_logits": disease_logits,
            "self_attn_weights": self_attn_weights,
            "asp_weights": asp_weights,
        }

class PEFTWavLM_Try1(nn.Module):
    def __init__(self, feature_dim, dropout, hidden_dim_classifier, output_dim, lora_rank, lora_alpha, target_modules, spk_dim, **kwargs):
        super(PEFTWavLM_Try1, self).__init__()

        from transformers import WavLMModel
        from peft import get_peft_model, LoraConfig, TaskType

        self.backbone_model = WavLMModel.from_pretrained(
            "microsoft/wavlm-large",
            output_hidden_states=True
        )
        self.backbone_model.freeze_feature_encoder()

        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=dropout,
            target_modules=target_modules,
        )
        self.backbone_model = get_peft_model(self.backbone_model, lora_config)

        self.pool = modules.AttentiveStatisticsPooling(channels=feature_dim)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim_classifier),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_classifier, output_dim)
        )

    def disabled_calc_additional_loss(self, embeddings, labels):
        # embeddings: [B, D]
        # lables: [B]
        if labels.dim() == 2:
            labels = torch.argmax(labels, dim=1)

        # -------------- contrastive_loss -------------------
        # temperature = 0.7
        # embeddings = F.normalize(embeddings, dim=1)
        # sim = torch.matmul(embeddings, embeddings.T) / temperature

        # labels = labels.unsqueeze(1)
        # mask = (labels == labels.T).float()

        # logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0), device=mask.device)
        # mask = mask * logits_mask

        # exp_sim = torch.exp(sim) * logits_mask
        # log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-9)

        # mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-9)
        # loss = -mean_log_prob_pos.mean()
        # return loss

    def get_feat_extract_output_lengths(self, input_length):
        """
        Computes the output length of the convolutional layers
        """
        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return (input_length - kernel_size) // stride + 1
        for kernel_size, stride in zip(self.backbone_model.config.conv_kernel, self.backbone_model.config.conv_stride):
            input_length = _conv_out_length(input_length, kernel_size, stride)
        return input_length

    def forward(self, x, attention_mask=None, **kwargs):
        x = x.squeeze(1)
        with torch.no_grad():
            x = self.backbone_model.feature_extractor(x)
            x = x.transpose(1, 2)  # New version of huggingface
            x, _ = self.backbone_model.feature_projection(
                x)  # New version of huggingface

        if attention_mask is not None:
            length = commons.compute_length_from_mask(
                attention_mask.detach().cpu())
            length = torch.tensor(length).cuda()

        x = self.backbone_model.encoder(
            x, output_hidden_states=True
        )  # .hidden_states
        features = x.last_hidden_state  # torch.Size([32, 24, 1024])

        feature_embedding, asp_weights = self.pool(features.permute(0, 2, 1))
        disease_logits = self.classifier(feature_embedding)

        return {
            "disease_logits": disease_logits,
            "embedding": feature_embedding,
            "asp_weights": asp_weights,
        }
    
class PEFTQwen3_Try1(nn.Module):
    def __init__(self, input_size, output_dim, lora_rank, lora_alpha, target_modules, spk_dim, **kwargs):
        super(PEFTQwen3_Try1, self).__init__()

        from peft import get_peft_model, LoraConfig, TaskType
        from transformers import Qwen3OmniMoeThinkerForConditionalGeneration

        # "target_modules": [ "q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2", "conv_out", "proj1", "proj2" ]
        temp_model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
            "/run/media/fourier/Data1/Pras/pretrain_models/Qwen3-Omni-30B-A3B-Thinking",
            torch_dtype="auto",
            device_map="cpu"
        )
        self.audio_tower = temp_model.audio_tower
        self.audio_tower.cuda()
        del temp_model
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        self.audiotower_hidden_dim = self.audio_tower.config.output_dim
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=0.1,
            target_modules=target_modules,
        )
        self.audio_tower = get_peft_model(self.audio_tower, lora_config)

        base_model = self.audio_tower.base_model.model
        orig_forward = base_model.forward

        TEXT_ONLY_KEYS = {
            "inputs_embeds",
            "labels",
            "attention_mask",
            "output_attentions",
            "output_hidden_states",
            "return_dict",
            "decoder_input_ids",
            "decoder_attention_mask",
        }

        def patched_forward(*args, **kwargs):
            if "input_ids" in kwargs and "input_features" not in kwargs:
                kwargs["input_features"] = kwargs.pop("input_ids")
            else:
                kwargs.pop("input_ids", None)

            for k in TEXT_ONLY_KEYS:
                kwargs.pop(k, None)
            return orig_forward(*args, **kwargs)
        base_model.forward = patched_forward

        self.pool = modules.AttentiveStatisticsPooling(channels=self.audiotower_hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(self.audiotower_hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_dim)
        )

    def after_cnn_len(self, L):
        L = (L - 1) // 2 + 1
        L = (L - 1) // 2 + 1
        L = (L - 1) // 2 + 1
        return L

    def forward(self, input_features, attention_mask=None, **kwargs):
        input_features = input_features.to(torch.bfloat16)
        feature_attention_mask = attention_mask.long()
        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
            input_features = input_features.permute(
                0, 2, 1)[feature_attention_mask.bool()].permute(1, 0)
        else:
            audio_feature_lengths = None

        feature_lens = audio_feature_lengths if audio_feature_lengths is not None else feature_attention_mask.sum(
            -1)
        audio_outputs = self.audio_tower(
            input_features,
            feature_lens=feature_lens,
        )
        audio_features = audio_outputs.last_hidden_state
        post_lens = torch.tensor(
            [self.after_cnn_len(l.item()) for l in feature_lens],
            device=feature_lens.device
        )

        total = audio_features.size(0)
        delta = total - post_lens.sum()
        if delta != 0:
            post_lens[-1] += delta

        audio_features = audio_features.split(post_lens.tolist(), dim=0)
        audio_features = pad_sequence(audio_features, batch_first=True) # for Attentive Pooling
        audio_features = audio_features.to(torch.float32)
        feature_embedding, asp_weights = self.pool(audio_features.permute(0, 2, 1))
        # audio_features = torch.stack([x.mean(dim=0)
        #                              for x in audio_features], dim=0)

        disease_logits = self.classifier(feature_embedding)

        return {
            "disease_logits": disease_logits,
            "asp_weights": asp_weights,
        }
    

class AST_Try1(nn.Module):
    def __init__(self, input_size, output_dim, lora_rank, lora_alpha, target_modules, spk_dim, **kwargs):
        super(AST_Try1, self).__init__()

        from transformers import ASTForAudioClassification
        temp_model = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.audio_tower = temp_model.audio_spectrogram_transformer
        self.audio_tower.cuda()
        self.model_config = temp_model.config
        del self.model_config.label2id
        del self.model_config.id2label
        del temp_model

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        self.layernorm = nn.LayerNorm(self.model_config.hidden_size, eps=self.model_config.layer_norm_eps)
        self.classifier = nn.Linear(self.model_config.hidden_size, 1)

    def forward(self, input_features, attention_mask=None, **kwargs):
        input_features = input_features.permute(0, 2, 1)
        pooled_output = self.audio_tower(input_values=input_features).pooler_output
        disease_logits = self.classifier(pooled_output)

        return {
            "disease_logits": disease_logits,
        }