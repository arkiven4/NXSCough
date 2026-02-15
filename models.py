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

class Res2NetVanilla(nn.Module):
    def __init__(self,
                 dummy_input,
                 feature_dim: int = 39,
                 embed_dim=192,
                 pooling_func='TSTP',
                 output_dim: int = 2, **kwargs):
        super(Res2NetVanilla, self).__init__()

        block = getattr(layers, "BasicBlockRes2Net")
        num_blocks = [3, 4, 6, 3]
        m_channels = 32

        self.in_planes = m_channels
        self.feature_dim = feature_dim

        def _downsample_freq(h: int, stages: int = 3) -> int:
            for _ in range(stages):
                h = (h - 1) // 2 + 1
            return h
        self.stats_dim = _downsample_freq(feature_dim) * m_channels * 8

        self.conv1 = nn.Conv2d(1,
                               m_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels)
        self.layer1 = self._make_layer(block,
                                       m_channels,
                                       num_blocks[0],
                                       stride=1)
        self.layer2 = self._make_layer(block,
                                       m_channels * 2,
                                       num_blocks[1],
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       m_channels * 4,
                                       num_blocks[2],
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       m_channels * 8,
                                       num_blocks[3],
                                       stride=2)

        self.pool = modules.TSTP(in_dim=self.stats_dim * block.expansion)
        self.pool_out_dim = self.pool.get_out_dim()

        self.classifier = nn.Sequential(
            nn.Linear(self.pool_out_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_dim)
        )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def get_frame_level_feat(self, x):
        # for outer interface
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # out = out.transpose(1, 3)
        # out = torch.flatten(out, 2, -1)
        return out

    def forward(self, x, **kwargs):
        """
        x: (B, n_mels, T) mel-spectrogram frames
        """
        x = x.unsqueeze_(1)
        x = self.get_frame_level_feat(x)

        stats = self.pool(x)

        disease_logits = self.classifier(stats)
        return {
            "disease_logits": disease_logits,
        }


class ResNet34ManualClassifier(nn.Module):
    def __init__(
        self,
        dummy_input,
        feature_dim: int = 39,
        output_dim: int = 2, **kwargs
    ):
        super().__init__()

        self.encoder1 = modules.Resnet34Manual(feature_dim=feature_dim)
        self.pool_out_dim = self.encoder1.pool.get_out_dim()

        self.classifier = nn.Sequential(
            nn.Linear(self.pool_out_dim, 128),
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

class ResNet34MultiEncoderClassifier(nn.Module):
    def __init__(
        self,
        dummy_input,
        feature_dim: int = 39,
        output_dim: int = 2, **kwargs
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.encoders = nn.ModuleList([
            modules.Resnet34Manual(feature_dim=feature_dim)
            for _ in range(3)
        ])
        self.pool_out_dim = self.encoders[0].pool.get_out_dim()

        self.fusion_logits = nn.Parameter(torch.zeros(3))
        self.classifier = nn.Sequential(
            nn.Linear(self.pool_out_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, output_dim)
        )

    def forward(self, x, tabular_ids=None, **kwargs):
        """
        x: (B, n_mels, T) mel-spectrogram frames
        """
        streams = torch.split(x, self.feature_dim, dim=1)
        feats = []
        for encoder, s in zip(self.encoders, streams):
            z = encoder(s.unsqueeze(1))   # (B, 64, T')
            feats.append(z)

        feats = torch.stack(feats, dim=1)      # (B, 3, 128)
        w = torch.softmax(self.fusion_logits[:feats.size(1)], dim=0)
        fused = torch.sum(feats * w[None, :, None], dim=1)
        
        disease_logits = self.classifier(fused)
        return {
            "disease_logits": disease_logits,
        }

class ResNet34HalfClassifier(nn.Module):
    def __init__(self, dummy_input,
        feature_dim: int = 39,
        output_dim: int = 2, **kwargs):
        super().__init__()

        block = modules.BasicBlock
        num_blocks = [3, 4, 6, 3]
        m_channels = 32

        def _downsample_freq(h: int, stages: int = 3) -> int:
            for _ in range(stages):
                h = (h - 1) // 2 + 1
            return h
        
        self.in_planes = m_channels
        self.feature_dim = feature_dim
        self.stats_dim = _downsample_freq(feature_dim) * m_channels * 8

        self.conv1 = nn.Conv2d(1,
                               m_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels)
        self.layer1 = self._make_layer(block,
                                       m_channels,
                                       num_blocks[0],
                                       stride=1)
        self.layer2 = self._make_layer(block,
                                       m_channels * 2,
                                       num_blocks[1],
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       m_channels * 4,
                                       num_blocks[2],
                                       stride=2)
        
        channel = 128
        self.se_channel = nn.Sequential(
            nn.Linear(channel, channel // 8),
            nn.ReLU(),
            nn.Linear(channel // 8, channel),
            nn.Sigmoid()
        )

        self.asp_freq = modules.AttentiveStatisticsPooling(channels=channel)
        self.asp_time = modules.AttentiveStatisticsPooling(channels=channel)

        self.classifier = nn.Sequential(
            nn.Linear(4 * channel, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, output_dim)
        )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)


    def forward(self, x, tabular_ids=None, **kwargs):
        """
        x: (B, n_mels, T) mel-spectrogram frames
        """
        x = x.unsqueeze_(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out) # [128, 64, 40, 28]
        out = self.layer3(out) # [128, 64, 40, 28]

        B, C, _, _ = out.shape
        pooled_c = out.mean(dim=(2, 3))          # [B, C]
        attn_c = self.se_channel(pooled_c)                # [B, C]
        out = out * attn_c.view(B, C, 1, 1)
        
        # Frequency ASP
        x_f = out.mean(dim=3)                    # [B, C, F]
        stats_f, attn_f = self.asp_freq(x_f)   # attn_f: [B, F]

        # Time ASP
        x_t = out.mean(dim=2)                    # [B, C, T]
        stats_t, attn_t = self.asp_time(x_t)   # attn_t: [B, T]

        # Fusion
        feats = torch.cat([stats_f, stats_t], dim=1)
        disease_logits = self.classifier(feats)

        return {
            "disease_logits": disease_logits,
            "attn_channel": attn_c,   # [B, C]
            "attn_freq": attn_f,      # [B, F]
            "attn_time": attn_t       # [B, T]
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

class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        dummy_input,
        feature_dim: int = 39,
        hidden_size: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_tabular: bool = False,
        fusion_type: str = "film",  # ["gating", "cross_attn", "film"]
        output_dim: int = 2, **kwargs
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )

        self.pool = modules.TemporalStatsPooling()
        self.audio_project = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),  # * 2 normal, *4 TSP
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256)
        )

        self.use_tabular = use_tabular
        if self.use_tabular:
            self.tabular_encoder = nn.Sequential(
                nn.Linear(4, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(32, 128),
                nn.ReLU(),
            )

            # Project tabular → audio space
            self.tabular_project = nn.Linear(128, 256)

            # Gate: decides how much tabular matters
            self.gate = nn.Sequential(
                nn.Linear(128, 256),
                nn.Sigmoid()
            )
            fusion_dim = 256
        else:
            fusion_dim = 256

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim)
        )

    def forward(self, x, tabular_ids=None, **kwargs):
        """
        x: (B, n_mels, T) mel-spectrogram frames
        """
        x = x.permute(0, 2, 1)
        audio_feat, _ = self.lstm(x)           # (B, T, 2H)
        
        audio_feat = self.pool(audio_feat)
        #w = F.softmax(self.pool(audio_feat), dim=1)   # (B, T, 1)
        #audio_feat = torch.sum(w * audio_feat, dim=1)
        
        # audio_feat = audio_feat.mean(dim=1)           # temporal pooling
        audio_feat = self.audio_project(audio_feat)      # (B, num_classes)

        if self.use_tabular:
            assert tabular_ids is not None

            tab_feat = self.tabular_encoder(tabular_ids)   # (B, 64)
            tab_proj = self.tabular_project(tab_feat)      # (B, 256)
            gate = self.gate(tab_feat)                     # (B, 256)

            fused = audio_feat + gate * tab_proj
        else:
            fused = audio_feat

        disease_logits = self.classifier(fused)
        return {
            "disease_logits": disease_logits,
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
                num_heads=att_head,
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
    def __init__(self, input_size, output_dim, lora_rank, lora_alpha, target_modules, spk_dim, **kwargs):
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
            lora_dropout=0.1,
            target_modules=target_modules,
        )
        self.backbone_model = get_peft_model(self.backbone_model, lora_config)

        self.pool = modules.AttentiveStatisticsPooling(channels=input_size)
        self.classifier = nn.Sequential(
            nn.Linear(input_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_dim)
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

class TemporalAttention(nn.Module):
    """
    Additive attention over time.
    Input: (B, T, D)
    Output: (B, D), (B, T)
    """
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, 1, bias=False)
        )

    def forward(self, x, mask=None):
        # x: (B, T, D)
        scores = self.attn(x).squeeze(-1)  # (B, T)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        weights = F.softmax(scores, dim=1)  # (B, T)
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)  # (B, D)
        return pooled, weights


class CNN_BiLSTM_Attention(nn.Module):
    def __init__(
        self,
        dummy_input,
        feature_dim: int = 39,
        hidden_size: int = 256,
        num_layers: int = 1,
        output_dim: int = 2, 
        cnn_channels=(1, 32, 64),
        pool_kernel=(2, 2),
        **kwargs
    ):
        super().__init__()

        # CNN
        self.conv1 = nn.Conv2d(cnn_channels[0], cnn_channels[1], 3, padding=1)
        self.conv2 = nn.Conv2d(cnn_channels[1], cnn_channels[2], 3, padding=1)
        self.pool = nn.MaxPool2d(pool_kernel)

        freq_after_pool = feature_dim // (pool_kernel[0] ** 2)
        cnn_out_dim = cnn_channels[2] * freq_after_pool

        # Dim reduction
        self.dense = nn.Linear(cnn_out_dim, hidden_size)

        # BiLSTM
        self.bilstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        # Attention pooling
        self.attention = TemporalAttention(hidden_size * 2)

        # Output head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size * 2, output_dim),
        )

    def forward(self, x, mask=None, **kwargs):
        """
        x: (B, n_mels, T)
        mask: (B, T') optional, after CNN pooling
        """
        x = x.unsqueeze(1)  # (B, 1, n_mels, T)

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # (B, C, F', T')

        x = x.permute(0, 3, 1, 2).contiguous()
        B, T, C, Freq = x.shape
        x = x.view(B, T, C * Freq)

        x = F.relu(self.dense(x))
        x, _ = self.bilstm(x)

        # Attention pooling over time
        x, attn_weights = self.attention(x, mask)

        disease_logits = self.classifier(x)
        return {
            "disease_logits": disease_logits,
            "attn_weights": attn_weights,
        }
