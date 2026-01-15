import torch, math
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

import torchvision
from transformers import AutoConfig, AutoFeatureExtractor

from typing import Any, Callable, Optional
from collections import OrderedDict

import modules, commons, layers
from torch.autograd import Function

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None

def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)

class GRL(nn.Module):
    def __init__(self, λ=1.0):
        super().__init__()
        self.λ = λ

    def forward(self, x):
        return GradReverse.apply(x, self.λ)
#################################################################################################
def make_pad_mask(lengths, max_len=None, device=None):
    if max_len is None:
        max_len = int(lengths.max().item())
    idxs = torch.arange(0, max_len, device=device).unsqueeze(0)  # [1, T]
    mask = (idxs < lengths.unsqueeze(1))  # [B, T]
    return mask

class MyOwnQwen_CrossAttent(nn.Module):
    def __init__(self, input_size, regress_hidden_dim, num_transformer_layers,
                 output_dim, spk_dim, **kwargs):
        super().__init__()

        from transformers import Qwen2_5OmniThinkerForConditionalGeneration
        temp_model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-Omni-3B",
            torch_dtype="auto",
            device_map="auto"
        )

        hidden_proj_dim = 512
        cross_heads = 8

        self.audio_tower = temp_model.audio_tower
        del temp_model
        import gc, torch
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        self.audiotower_hidden_dim = self.audio_tower.config.output_dim
        self.project_shared = nn.Linear(self.audiotower_hidden_dim, hidden_proj_dim)

        # Branch projections
        self.spk_proj = nn.Linear(hidden_proj_dim, hidden_proj_dim)
        self.gender_proj = nn.Linear(hidden_proj_dim, hidden_proj_dim)
        self.dis_proj = nn.Linear(hidden_proj_dim, hidden_proj_dim)

        self.cross = modules.BidirectionalCrossAttention(hidden_proj_dim, num_heads=cross_heads)

        # TOKEN-LEVEL CE HEADS
        self.spk_head = nn.Linear(hidden_proj_dim, spk_dim)
        self.gender_head = nn.Linear(hidden_proj_dim, 2)
        self.dis_head = nn.Linear(hidden_proj_dim, output_dim)

        # Utterance-level heads
        self.utt_speaker_cls = nn.Linear(hidden_proj_dim, spk_dim)
        self.utt_gender_cls = nn.Linear(hidden_proj_dim, 2)
        self.utt_disease_cls = nn.Linear(hidden_proj_dim, output_dim)

        self.kl_weight = 0.1


    def forward(self, input_features, attention_mask=None, grl_lambda=0.0):
        device = input_features.device
        feature_attention_mask = attention_mask.long()

        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
            input_features = input_features.permute(0, 2, 1)[feature_attention_mask.bool()].permute(1, 0)
        else:
            audio_feature_lengths = None

        feature_lens = audio_feature_lengths if audio_feature_lengths is not None else feature_attention_mask.sum(-1)
        audio_outputs = self.audio_tower(
            input_features,
            feature_lens=feature_lens,
        )
        audio_features = audio_outputs.last_hidden_state

        # audio_feat_lengths, audio_output_lengths = self.audio_tower._get_feat_extract_output_lengths(
        #     audio_feature_lengths
        # )

        # x = self.audio_tower(
        #     input_features=input_features,
        #     feature_lens=audio_feature_lengths,
        #     aftercnn_lens=audio_feat_lengths
        # ).last_hidden_state

        # x = x.split(audio_output_lengths.tolist(), dim=0)
        # x = pad_sequence(x, batch_first=True)

        # z = self.project_shared(x)
        # mask = make_pad_mask(audio_output_lengths, max_len=z.size(1), device=device)

        # # latent streams
        # spk_latent = self.spk_proj(z)
        # gender_latent = self.gender_proj(z)
        # dis_latent = self.dis_proj(z)

        # spk_stream = spk_latent + gender_latent
        # spk_ctx, dis_ctx = self.cross(spk_stream, dis_latent, a_mask=mask, b_mask=mask)

        # # TOKEN-LEVEL EMISSIONS
        # spk_emissions = self.spk_head(spk_ctx)         # [B,T,spk_dim]
        # gender_emissions = self.gender_head(spk_ctx)   # [B,T,2]
        # dis_emissions = self.dis_head(dis_ctx)         # [B,T,out_dim]

        # outputs = {}
        # total_loss = 0.0

        # # KL ALIGNMENT
        # spk_logp = F.log_softmax(spk_emissions, -1)
        # dis_prob = F.softmax(dis_emissions, -1)

        # shared_dim = min(spk_logp.size(-1), dis_prob.size(-1))
        # spk_proj = spk_logp[..., :shared_dim]
        # dis_proj = dis_prob[..., :shared_dim]

        # kl_sym = 0.5 * (
        #     F.kl_div(spk_proj, dis_proj, reduction='batchmean') +
        #     F.kl_div(torch.log(dis_proj+1e-12), torch.exp(spk_proj), reduction='batchmean')
        # )

        # total_loss += self.kl_weight * kl_sym
        # outputs['kl_sym'] = kl_sym

        # # POOLING HEADS
        # mask_f = mask.float().unsqueeze(-1)
        # pooled_spk = (spk_ctx * mask_f).sum(1) / (mask_f.sum(1)+1e-9)
        # pooled_dis = (dis_ctx * mask_f).sum(1) / (mask_f.sum(1)+1e-9)

        # outputs['speaker_logits'] = self.utt_speaker_cls(pooled_spk)
        # outputs['gender_logits']  = self.utt_gender_cls(pooled_spk)
        # outputs['disease_logits'] = self.utt_disease_cls(pooled_dis)

        # outputs["internal_loss"] = total_loss

        return outputs

    
class MyOwnQwen_AudioEncoder(nn.Module):
    def __init__(self, input_size, regress_hidden_dim, num_transformer_layers, output_dim, spk_dim, **kwargs):
        super(MyOwnQwen_AudioEncoder, self).__init__()

        from transformers import Qwen2_5OmniThinkerForConditionalGeneration
        temp_model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-Omni-3B",
            torch_dtype="auto",
            device_map="auto"
        )
        self.audio_tower = temp_model.audio_tower
        self.audiotower_hidden_dim = self.audio_tower.config.output_dim

        self.pooling = layers.AttentiveStatisticsPooling(self.audiotower_hidden_dim, attention_dim=1024)
        embed_dim = self.audiotower_hidden_dim * 2  # Since ASP outputs 2*input_size
        dropout = 0.2
        
        # Projection layers for multi-task learning
        self.disease_clf = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, output_dim)
        )


    def forward(self, input_features, attention_mask=None, grl_lambda=0.0):
        attention_mask = attention_mask.long()

        audio_feature_lengths = torch.sum(attention_mask, dim=1)
        input_features = input_features.permute(0, 2, 1)[attention_mask.bool()].permute(1, 0)

        audio_feat_lengths, audio_output_lengths = self.audio_tower._get_feat_extract_output_lengths(
            audio_feature_lengths if audio_feature_lengths is not None else attention_mask.sum(-1))

        feature_lens = audio_feature_lengths if audio_feature_lengths is not None else attention_mask.sum(-1)

        x = self.audio_tower(
            input_features=input_features,
            feature_lens=feature_lens,
            aftercnn_lens=audio_feat_lengths
        )

        x = x.last_hidden_state
        if x.shape[0] != sum(audio_output_lengths.tolist()):
            raise ValueError("length of audio_features should match audio_output_lengths")

        x = x.split(audio_output_lengths.tolist(), dim=0)
        x = pad_sequence(x, batch_first=True)  # [batch, max_len, hidden_dim]

        feature_embedding = self.pooling(x)
        disease_logits = self.disease_clf(feature_embedding)
        return {
            "disease_logits": disease_logits,
        }
    
class PEFTWavlm_MyOwnGradeReversal(nn.Module):
    def __init__(self, input_size, regress_hidden_dim, num_transformer_layers, output_dim, spk_dim, **kwargs):
        super(PEFTWavlm_MyOwnGradeReversal, self).__init__()

        self.pooling = layers.AttentiveStatisticsPooling(input_size, attention_dim=128)
        embed_dim = input_size * 2  # Since ASP outputs 2*input_size
        dropout = 0.1

        # Projection layers for multi-task learning
        self.disease_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),  # Use GELU like WavLM
            nn.LayerNorm(embed_dim // 2, eps=1e-5),
            nn.Dropout(dropout)
        )
        self.speaker_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.LayerNorm(embed_dim // 2, eps=1e-5),
            nn.Dropout(dropout)
        )

        # Classifiers
        self.disease_clf = nn.Sequential(
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 4, output_dim)
        )

        self.speaker_clf = nn.Sequential(
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 4, spk_dim)
        )
        
        self.gender_clf = nn.Sequential(
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 4, 2)
        )


    def forward(self, x, attention_mask=None, grl_lambda=0.0):
        feature_embedding = self.pooling(x)
        
        d_emb = self.disease_proj(feature_embedding)    
        s_emb = self.speaker_proj(feature_embedding) 

        disease_logits = self.disease_clf(d_emb)

        s_in = grad_reverse(s_emb, grl_lambda) if self.training and grl_lambda > 0 else s_emb
        speaker_logits = self.speaker_clf(s_in)
        gender_logits = self.gender_clf(s_in)

        return {
            "disease_logits": disease_logits,
            "speaker_logits": speaker_logits,
            "gender_logits": gender_logits,
            "d_emb": d_emb,
            "s_emb": s_emb,
        }

class WavLMEncoder_MyOwn(nn.Module):
    def __init__(self, input_size, regress_hidden_dim, num_transformer_layers, output_dim, spk_dim, **kwargs):
        super(WavLMEncoder_MyOwn, self).__init__()

        config = AutoConfig.from_pretrained("microsoft/wavlm-large")
        self.feature_extractor = layers.WavLMFeatureEncoder(config)
        self.feature_projection = layers.WavLMFeatureProjection(config)

        embed_dim = 1024
        num_heads = 16
        dropout = 0.1

        self.pos_conv_embed = nn.Conv1d(
            embed_dim, embed_dim, kernel_size=128, padding=64, groups=16
        )
        self.layer_norm = nn.LayerNorm(embed_dim, eps=1e-5)

        self.attention_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,  # 4096 for WavLM-large
                dropout=dropout,
                activation='gelu',  # WavLM uses GELU
                layer_norm_eps=1e-5,
                batch_first=True,
                norm_first=True  # Pre-norm like WavLM
            ) for _ in range(num_transformer_layers)
        ])
        self.final_layer_norm = nn.LayerNorm(embed_dim, eps=1e-5)
        self.layer_weights = nn.Parameter(torch.ones(num_transformer_layers + 1))
        self.pooling = nn.AdaptiveAvgPool1d(1)

        # Classifiers
        self.disease_clf = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, output_dim)
        )


    def forward(self, x, attention_mask=None, grl_lambda=0.0):
        extract_features = self.feature_extractor(x)
        extract_features = extract_features.transpose(1, 2)
        hidden_states, extract_features = self.feature_projection(extract_features) # B, T, D

        pos_conv_output = self.pos_conv_embed(hidden_states.transpose(1, 2))
        pos_conv_output = pos_conv_output.transpose(1, 2)
        min_seq_len = min(hidden_states.size(1), pos_conv_output.size(1))
        hidden_states = hidden_states[:, :min_seq_len, :]
        pos_conv_output = pos_conv_output[:, :min_seq_len, :]

        hidden_states = hidden_states + pos_conv_output
        hidden_states = self.layer_norm(hidden_states)

        layer_outputs = [hidden_states]

        for layer in self.attention_layers:
            hidden_states = layer(hidden_states)
            layer_outputs.append(hidden_states)
        
        hidden_states = self.final_layer_norm(hidden_states)
        layer_outputs[-1] = hidden_states
        layer_weights_normalized = torch.softmax(self.layer_weights, dim=0)
        weighted_hidden_states = torch.zeros_like(hidden_states)
        for i, layer_output in enumerate(layer_outputs):
            weighted_hidden_states += layer_weights_normalized[i] * layer_output
        
        hidden_states = weighted_hidden_states.transpose(1, 2)  # [B, embed_dim, T]
        feature_embedding = self.pooling(hidden_states).squeeze(-1)  # [B, embed_dim]
        
        disease_logits = self.disease_clf(feature_embedding)

        return {
            "disease_logits": disease_logits,
        }
    
class WavLMEncoder_MyOwnGradeReversal(nn.Module):
    def __init__(self, input_size, regress_hidden_dim, num_transformer_layers, output_dim, spk_dim, **kwargs):
        super(WavLMEncoder_MyOwnGradeReversal, self).__init__()

        config = AutoConfig.from_pretrained("microsoft/wavlm-large")
        self.feature_extractor = layers.WavLMFeatureEncoder(config)
        self.feature_projection = layers.WavLMFeatureProjection(config)

        embed_dim = 1024
        num_heads = 16
        dropout = 0.1

        self.pos_conv_embed = nn.Conv1d(
            embed_dim, embed_dim, kernel_size=128, padding=64, groups=16
        )
        self.layer_norm = nn.LayerNorm(embed_dim, eps=1e-5)

        self.attention_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,  # 4096 for WavLM-large
                dropout=dropout,
                activation='gelu',  # WavLM uses GELU
                layer_norm_eps=1e-5,
                batch_first=True,
                norm_first=True  # Pre-norm like WavLM
            ) for _ in range(num_transformer_layers)
        ])
        self.final_layer_norm = nn.LayerNorm(embed_dim, eps=1e-5)
        self.layer_weights = nn.Parameter(torch.ones(num_transformer_layers + 1))
        self.pooling = nn.AdaptiveAvgPool1d(1)

        # Projection layers for multi-task learning
        self.disease_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),  # Use GELU like WavLM
            nn.LayerNorm(embed_dim // 2, eps=1e-5),
            nn.Dropout(dropout)
        )
        self.speaker_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.LayerNorm(embed_dim // 2, eps=1e-5),
            nn.Dropout(dropout)
        )

        # Classifiers
        self.disease_clf = nn.Sequential(
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 4, output_dim)
        )

        self.speaker_clf = nn.Sequential(
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 4, spk_dim)
        )
        
        self.gender_clf = nn.Sequential(
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 4, 2)
        )


    def forward(self, x, attention_mask=None, grl_lambda=0.0):
        extract_features = self.feature_extractor(x)
        extract_features = extract_features.transpose(1, 2)
        hidden_states, extract_features = self.feature_projection(extract_features) # B, T, D

        pos_conv_output = self.pos_conv_embed(hidden_states.transpose(1, 2))
        pos_conv_output = pos_conv_output.transpose(1, 2)
        min_seq_len = min(hidden_states.size(1), pos_conv_output.size(1))
        hidden_states = hidden_states[:, :min_seq_len, :]
        pos_conv_output = pos_conv_output[:, :min_seq_len, :]

        hidden_states = hidden_states + pos_conv_output
        hidden_states = self.layer_norm(hidden_states)

        layer_outputs = [hidden_states]

        for layer in self.attention_layers:
            hidden_states = layer(hidden_states)
            layer_outputs.append(hidden_states)
        
        hidden_states = self.final_layer_norm(hidden_states)
        layer_outputs[-1] = hidden_states
        layer_weights_normalized = torch.softmax(self.layer_weights, dim=0)
        weighted_hidden_states = torch.zeros_like(hidden_states)
        for i, layer_output in enumerate(layer_outputs):
            weighted_hidden_states += layer_weights_normalized[i] * layer_output
        
        hidden_states = weighted_hidden_states.transpose(1, 2)  # [B, embed_dim, T]
        feature_embedding = self.pooling(hidden_states).squeeze(-1)  # [B, embed_dim]
        
        d_emb = self.disease_proj(feature_embedding)    
        s_emb = self.speaker_proj(feature_embedding) 

        disease_logits = self.disease_clf(d_emb)

        s_in = grad_reverse(s_emb, grl_lambda) if self.training and grl_lambda > 0 else s_emb
        speaker_logits = self.speaker_clf(s_in)
        gender_logits = self.gender_clf(s_in)

        return {
            "disease_logits": disease_logits,
            "speaker_logits": speaker_logits,
            "gender_logits": gender_logits,
            "d_emb": d_emb,
            "s_emb": s_emb,
        }

class SE_Trans(nn.Module):
    def __init__(
        self, input_size, output_dim, spk_dim, **kwargs
    ):
        """
        :param enable_multimodal: enable multimodal ASC
        :param num_locations: number of city
        :param location_embedding_dim: city embedding dim
        :param time_feature_dim: time feature dim
        :param time_mapping_dim: time embedding dim
        """
        super(SE_Trans, self).__init__()

        nhead = 8
        dim_feedforward = 32
        n_layers = 1
        dropout = 0.1
        embed_dim = 128
        
        self.SE_block1 = layers.SEBlock(in_channels=1, out_channels=64)
        self.SE_block2 = layers.SEBlock(in_channels=64, out_channels=embed_dim)
        self.global_pool = nn.AdaptiveAvgPool2d((16, 1))
        
        self.encoder_layer = nn.TransformerEncoderLayer(
            embed_dim, nhead, dim_feedforward, dropout
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, n_layers)
        
        self.fc = nn.Linear(embed_dim, output_dim, bias=True)
        
        self.bn0 = nn.BatchNorm2d(input_size)
        self.init_weights()

        # Projection layers for multi-task learning
        self.disease_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),  # Use GELU like WavLM
            nn.LayerNorm(embed_dim // 2, eps=1e-5),
            nn.Dropout(dropout)
        )
        self.speaker_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.LayerNorm(embed_dim // 2, eps=1e-5),
            nn.Dropout(dropout)
        )

        # Classifiers
        self.disease_clf = nn.Sequential(
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 4, output_dim)
        )

        self.speaker_clf = nn.Sequential(
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 4, spk_dim)
        )
        
        self.gender_clf = nn.Sequential(
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 4, 2)
        )

    def init_weights(self):
        modules.init_bn(self.bn0)
        modules.init_layer(self.fc)

    def forward(self, x, attention_mask=None, grl_lambda=0.0):
        """
        :param x: audio features [batch, frames, bins, 1]
        :param locations: city [batch]
        :param time_features: time feature [batch, 8]
        """
        x = x.unsqueeze(-1)
        #print(x.shape)
        #x = x.transpose(1, 3)  # x = [batch, bins, frames, in_chs]
        x = self.bn0(x)  # BN is done over the bins dimension
        x = x.transpose(1, 3)  # x = [batch, in_chs, frames, bins]

        x = self.SE_block1(x, pool_size=(2, 2), pool_type="avg")
        x = self.SE_block2(x, pool_size=(2, 2), pool_type="avg")

        x = self.global_pool(x)
        x = x.view(x.size(0), -1, x.size(2))  # x = [batch, in_chs, frames]
        x = x.permute(2, 0, 1)  # x = [frames, batch, in_chs]
        x = self.encoder(x)  # x = [frames, batch, in_chs]
        x = x.permute(1, 0, 2)  # x = [batch, frames, in_chs]
        (x, _) = torch.max(x, dim=1)  # (batch_size, in_chs=128)

        d_emb = self.disease_proj(x)    
        s_emb = self.speaker_proj(x) 

        disease_logits = self.disease_clf(d_emb)
        s_in = grad_reverse(s_emb, grl_lambda) if self.training and grl_lambda > 0 else s_emb
        speaker_logits = self.speaker_clf(s_in)
        gender_logits = self.gender_clf(s_in)

        return {
            "disease_logits": disease_logits,
            "speaker_logits": speaker_logits,
            "gender_logits": gender_logits,
            "d_emb": d_emb,
            "s_emb": s_emb,
        }

from torchvision.models import resnet50


class ResNet50Mel(nn.Module):
    def __init__(self,
        input_size,
        output_dim,
        spk_dim,
        embed_dim=2048,
        dropout=0.3,
        grl_lambda=1.0,
        **kwargs): 
        super().__init__()
        
        base = resnet50(pretrained=True)

        # Replace first conv to handle 1-channel input (mel)
        base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,
        )

        # Global pooling handles variable time dimension
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        #self.global_pool = layers.AttentiveStatisticsPooling(embed_dim, attention_dim=1024)
        #embed_dim = embed_dim * 2

        self.disease_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, output_dim),
        )


        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, attention_mask=None, grl_lambda=0.0):
        """
        x: [B, 1, F, T], T can vary per batch element
        """
        x = x.unsqueeze(1)
        feat = self.backbone(x)                # [B, 2048, F', T']
        feat = F.adaptive_avg_pool2d(feat, (feat.size(2), 1))
        pooled = self.global_pool(feat).flatten(1)

        # Attentive
        #feat = feat.squeeze(-1).permute(0, 2, 1)
        #pooled = self.global_pool(feat)#.flatten(1)  # [B, 2048]

        disease_logits = self.disease_head(pooled)

        return {
            "disease_logits": disease_logits,
        }
    
class ResNet50MelAdversarial(nn.Module):
    def __init__(self,
        input_size,
        output_dim,
        spk_dim,
        embed_dim=2048,
        dropout=0.3,
        grl_lambda=1.0,
        **kwargs): 
        super().__init__()
        
        base = resnet50(pretrained=True)

        # Replace first conv to handle 1-channel input (mel)
        base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,
        )

        # Global pooling handles variable time dimension
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        #self.global_pool = layers.AttentiveStatisticsPooling(embed_dim, attention_dim=1024)
        #embed_dim = embed_dim * 2

        # Shared projection
        self.disease_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),  # Use GELU like WavLM
            nn.LayerNorm(embed_dim // 2, eps=1e-5),
            nn.Dropout(dropout)
        )
        # self.speaker_proj = nn.Sequential(
        #     nn.Linear(embed_dim, embed_dim // 2),
        #     nn.GELU(),
        #     nn.LayerNorm(embed_dim // 2, eps=1e-5),
        #     nn.Dropout(dropout)
        # )

        # Disease head (normal)
        self.disease_head = nn.Sequential(
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 4, output_dim),
        )

        # Speaker head (adversarial)
        # self.speaker_head = nn.Sequential(
        #     nn.Linear(embed_dim // 2, embed_dim // 4),
        #     nn.GELU(),
        #     nn.Dropout(0.7),
        #     nn.Linear(embed_dim // 4, spk_dim),
        # )
        # self.gender_head = nn.Sequential(
        #     nn.Linear(embed_dim // 2, embed_dim // 4),
        #     nn.GELU(),
        #     nn.Dropout(0.7),
        #     nn.Linear(embed_dim // 4, 2),
        # )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, attention_mask=None, grl_lambda=0.0):
        """
        x: [B, 1, F, T], T can vary per batch element
        """
        x = x.unsqueeze(1)
        feat = self.backbone(x)                # [B, 2048, F', T']
        feat = F.adaptive_avg_pool2d(feat, (feat.size(2), 1))
        pooled = self.global_pool(feat).flatten(1)

        # Attentive
        #feat = feat.squeeze(-1).permute(0, 2, 1)
        #pooled = self.global_pool(feat)#.flatten(1)  # [B, 2048]

        d_emb = self.disease_proj(pooled)    
        disease_logits = self.disease_head(d_emb)

        # rev_feat = grad_reverse(pooled, grl_lambda) if self.training and grl_lambda > 0 else pooled
        # s_emb = self.speaker_proj(rev_feat)
        # speaker_logits = self.speaker_head(s_emb)
        # gender_logits = self.gender_head(s_emb)

        return {
            "disease_logits": disease_logits,
            # "speaker_logits": speaker_logits,
            # "gender_logits": gender_logits,
            # "d_emb": d_emb,
            # "s_emb": s_emb,
        }
    
import torch
import torch.nn as nn
from timm.models.swin_transformer import SwinTransformer

class SwinCoughClassifier(nn.Module):
    def __init__(self, input_size, output_dim=2, pretrained=True, **kwargs):
        super().__init__()
        # Load Swin backbone from timm
        self.backbone = SwinTransformer(
            img_size=224,            # adjust if your spectrogram is larger/smaller
            patch_size=4,
            in_chans=1,              # spectrograms are single-channel
            embed_dim=96,
            depths=(2, 2, 6, 2),
            num_heads=(3, 6, 12, 24),
            window_size=7,
            drop_path_rate=0.2,
            num_classes=0,           # remove classifier head
            pretrained_window_sizes=[7, 7, 7, 7] if pretrained else None
        )

        # Adaptive pooling for variable-length input
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, output_dim)
        )

    def forward(self, x, attention_mask=None, grl_lambda=0.0):
        # x: [B, 1, H, W]
        x = x.unsqueeze(1)
        features = self.backbone.forward_features(x)  # [B, C, H', W']
        features = features.mean(dim=[1, 2])          # global average pool
        disease_logits = self.classifier(features)

        return {
            "disease_logits": disease_logits,
            # "speaker_logits": speaker_logits,
            # "gender_logits": gender_logits,
            # "d_emb": d_emb,
            # "s_emb": s_emb,
        }

from torchvision.models.vision_transformer import VisionTransformer, vit_b_16
import torch.nn.functional as F
class ViTMel(nn.Module):
    def __init__(
        self,
        input_size,
        output_dim,
        spk_dim,
        embed_dim: int = 768,
        patch_size: int = 16,
        grl_lambda: float = 1.0,
        dropout: float = 0.3,
        **kwargs
    ):
        super().__init__()

        # Use torchvision's ViT backbone (requires fixed-size patch, but variable time is fine with padding)
        self.vit = vit_b_16(pretrained=True)
        # Modify first conv layer for 1-channel input
        self.vit.conv_proj = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)

        # Shared bottleneck
        self.disease_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.LayerNorm(embed_dim // 2),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, output_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, attention_mask=None, grl_lambda=0.0):
        """
        x: [B, 1, F, T] (mel-spectrogram, variable T)
        """
        x = x.unsqueeze(1)
        _, _, _, T = x.shape
        pad_T = (16 - (T % 16)) % 16
        if pad_T > 0:
            x = nn.functional.pad(x, (0, pad_T))

        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        vit_out = self.vit._process_input(x)

        # Extract tokens and class embedding
        n = vit_out.shape[0]
        cls_token = self.vit.class_token.expand(n, -1, -1)
        vit_out = torch.cat((cls_token, vit_out), dim=1)
        vit_out = self.vit.encoder(vit_out)
        pooled = vit_out[:, 0]  # CLS token as global embedding

        disease_logits = self.disease_head(pooled)
        
        return {
            "disease_logits": disease_logits,
        }
    
class ViTMelAdversarial(nn.Module):
    def __init__(
        self,
        input_size,
        output_dim,
        spk_dim,
        embed_dim: int = 768,
        patch_size: int = 16,
        grl_lambda: float = 1.0,
        dropout: float = 0.3,
        **kwargs
    ):
        super().__init__()

        # Use torchvision's ViT backbone (requires fixed-size patch, but variable time is fine with padding)
        self.vit = vit_b_16(pretrained=True)
        # Modify first conv layer for 1-channel input
        self.vit.conv_proj = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)

        # Shared bottleneck
        self.bottleneck = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.LayerNorm(embed_dim // 2),
            nn.Dropout(dropout),
        )

        # Disease head (normal)
        self.disease_head = nn.Sequential(
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 4, output_dim),
        )

        # Speaker head (adversarial)
        self.grl = GRL(λ=grl_lambda)
        self.speaker_head = nn.Sequential(
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 4, spk_dim),
        )
        self.gender_head = nn.Sequential(
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 4, 2),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, attention_mask=None, grl_lambda=0.0):
        """
        x: [B, 1, F, T] (mel-spectrogram, variable T)
        """
        x = x.unsqueeze(1)
        _, _, _, T = x.shape
        pad_T = (16 - (T % 16)) % 16
        if pad_T > 0:
            x = nn.functional.pad(x, (0, pad_T))

        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        vit_out = self.vit._process_input(x)

        # Extract tokens and class embedding
        n = vit_out.shape[0]
        cls_token = self.vit.class_token.expand(n, -1, -1)
        vit_out = torch.cat((cls_token, vit_out), dim=1)
        vit_out = self.vit.encoder(vit_out)
        pooled = vit_out[:, 0]  # CLS token as global embedding

        embed = self.bottleneck(pooled)

        disease_logits = self.disease_head(embed)
        adv_embed = self.grl(embed)
        speaker_logits = self.speaker_head(adv_embed)
        gender_logits = self.gender_head(adv_embed)

        return {
            "disease_logits": disease_logits,
            "speaker_logits": speaker_logits,
            "gender_logits": gender_logits
        }

from s3prl.upstream.mockingjay.builder import PretrainedTransformer
class TERA_TryDownstream(nn.Module):
    def __init__(self, input_size, output_dim, spk_dim, **kwargs):
        super(TERA_TryDownstream, self).__init__()

        options = {
            "load_pretrain": "True",
            "no_grad": "True",
            "dropout": "default",
            "spec_aug": "False",
            "spec_aug_prev": "False",
            "output_hidden_states": "True",
            "permute_input": "False",
        }
        options["ckpt_file"] = "pretrained/tera_pretrained.pth"
        options["select_layer"] = -1

        pretrained_dict = torch.load("pretrained/tera_pretrained.pth", weights_only=False)
        transformer_state = pretrained_dict['Transformer']

        self.tera_model = PretrainedTransformer(options, inp_dim=-1)
        self.tera_model.model.load_state_dict(transformer_state, strict=True)
        self.tera_model.eval()

        dropout = 0.1
        embed_dim = 768 * 2

        self.pooling = layers.AttentiveStatisticsPooling(embed_dim // 2, attention_dim=embed_dim // 4)

        # Projection layers for multi-task learning
        self.disease_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),  # Use GELU like WavLM
            nn.LayerNorm(embed_dim // 2, eps=1e-5),
            nn.Dropout(dropout)
        )
        self.speaker_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.LayerNorm(embed_dim // 2, eps=1e-5),
            nn.Dropout(dropout)
        )

        # Classifiers
        self.disease_clf = nn.Sequential(
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 4, output_dim)
        )

        self.speaker_clf = nn.Sequential(
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 4, spk_dim)
        )
        
        self.gender_clf = nn.Sequential(
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 4, 2)
        )

    def forward(self, x, attention_mask=None, grl_lambda=0.0):
        with torch.no_grad():
            x = self.tera_model(x)[0] # torch.Size([128, 51, 768])
        
        x = torch.nan_to_num(x, nan=0.0)
        feature_embedding = self.pooling(x)

        d_emb = self.disease_proj(feature_embedding)    
        s_emb = self.speaker_proj(feature_embedding) 

        disease_logits = self.disease_clf(d_emb)
        s_in = grad_reverse(s_emb, grl_lambda) if self.training and grl_lambda > 0 else s_emb
        speaker_logits = self.speaker_clf(s_in)
        gender_logits = self.gender_clf(s_in)

        return {
            "disease_logits": disease_logits,
            "speaker_logits": speaker_logits,
            "gender_logits": gender_logits,
            "d_emb": d_emb,
            "s_emb": s_emb,
        }
    
class Eff_MyOwn1(nn.Module):
    def __init__(self, input_size, regress_hidden_dim, output_dim, **kwargs):
        super(Eff_MyOwn1, self).__init__()

        self.cnn1 = torch.nn.Conv2d(1, 3, kernel_size=3)
        #self.efficientnet = EfficientNet.from_pretrained('efficientnet-b5') #
        self.efficientnet = EfficientNet.from_name("efficientnet-b0", include_top=False, drop_connect_rate=0.1) # [128, 1280, 1, 1]

        self.dense = nn.Linear(1280, 512)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(512, output_dim)


    def forward(self, x, attention_mask=None):
        x = x.unsqueeze(1) # [128, 1, 94, 64]
        x = self.cnn1(x) # [128, 3, 92, 62]
        x = self.efficientnet(x).squeeze(-1).squeeze(-1)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        disease_logits = self.out_proj(x)
        
        return {
            "disease_logits": disease_logits,
        }

######################################## MAE #########################################################

# model = SupervisedMaskedAutoencoderViT(
#         patch_size=16, embed_dim=768, depth=12, num_heads=12,
#         decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
#         mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

from functools import partial
from timm.models.vision_transformer import PatchEmbed, Block

class SupervisedMaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 cls_hidden_mlp=3072, output_dim=1000, global_pool=True,
                 mlp_depth=2, **kwargs):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # Classification branch specifics
        if cls_hidden_mlp == 0:
            self.cls_head = nn.Linear(embed_dim, output_dim)
        else:
            assert mlp_depth in [2], "mlp depth should be 2"
            if mlp_depth == 2:
                self.cls_head = nn.Sequential(
                    nn.Linear(embed_dim, cls_hidden_mlp),
                    nn.BatchNorm1d(cls_hidden_mlp),
                    nn.ReLU(inplace=True),
                    nn.Linear(cls_hidden_mlp, output_dim),
                )
        self.global_pool = global_pool
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = modules.get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = modules.get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_classification(self, x):
        if self.global_pool:
            feat = x[:, 1:, :].mean(dim=1)  # global pool without cls token
        else:
            feat = x[:, 0, :]  # with cls token
        logits = self.cls_head(feat)
        return logits

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss, self.unpatchify(pred)

    def forward(self, imgs, mask_ratio=0.75, attention_mask=None, grl_lambda=0.0):
        imgs = imgs.unsqueeze(1).repeat(1, 3, 1, 1)
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        # Reconstruction branch
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        # Classification branch
        logits = self.forward_classification(latent)
        loss, x_rec = self.forward_loss(imgs, pred, mask)

        return {
            "disease_logits": logits,
            "x_rec": x_rec.mean(dim=1),
            "internal_loss": loss
        }

#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################

# class LSTMAudioClassifier1(nn.Module):
#     def __init__(self, input_size, regress_hidden_dim, output_dim, **kwargs):
#         super(LSTMAudioClassifier1, self).__init__()
        
#         self.batch_norm1 = nn.BatchNorm1d(input_size)
#         self.lstm1 = nn.LSTM(input_size, regress_hidden_dim, batch_first=True)
        
#         self.batch_norm2 = nn.BatchNorm1d(regress_hidden_dim)
#         self.lstm2 = nn.LSTM(regress_hidden_dim, regress_hidden_dim, batch_first=True)
        
#         #self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8, batch_first=True)

#         self.flatten = nn.Flatten()
#         self.dropout = nn.Dropout(0.1)
#         self.fc = nn.Linear(regress_hidden_dim, output_dim)

#     def forward(self, x, lengths=None):
#         x = x.permute(0, 2, 1)
#         x = self.batch_norm1(x.transpose(1, 2)).transpose(1, 2)
#         x, _ = self.lstm1(x)
        
#         x = self.batch_norm2(x.transpose(1, 2)).transpose(1, 2)
#         x, _ = self.lstm2(x)
        
#         # attn_output, _ = self.attention(x, x, x)
#         # x = torch.mean(attn_output, dim=1)

#         x = self.flatten(x[:, -1, :])
#         x = self.dropout(x)

#         x = self.fc(x)

#         loss_internal = 0.0
#         return x #[x, embedding, loss_internal]



# class LSTMClassifierMT(nn.Module):
#     def __init__(self, input_size, regress_hidden_dim, output_dim, spk_dim, **kwargs):
#         super(LSTMClassifierMT, self).__init__()
        
#         self.batch_norm1 = nn.BatchNorm1d(input_size)
#         self.lstm1 = nn.LSTM(input_size, regress_hidden_dim, batch_first=True)
        
#         self.batch_norm2 = nn.BatchNorm1d(regress_hidden_dim)
#         self.lstm2 = nn.LSTM(regress_hidden_dim, regress_hidden_dim, batch_first=True)
        
#         self.flatten = nn.Flatten()

#         self.disease_proj = nn.Sequential(
#             nn.Linear(regress_hidden_dim, regress_hidden_dim // 2),
#             nn.ReLU(),
#             nn.LayerNorm(regress_hidden_dim // 2)
#         )
#         self.speaker_proj = nn.Sequential(
#             nn.Linear(regress_hidden_dim, regress_hidden_dim // 2),
#             nn.ReLU(),
#             nn.LayerNorm(regress_hidden_dim // 2)
#         )

#         self.disease_clf = nn.Sequential(
#             nn.Linear(regress_hidden_dim // 2, regress_hidden_dim // 4),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(regress_hidden_dim // 4, output_dim)
#         )

#         # speaker classifier receives GRL-wrapped features during training
#         self.speaker_clf = nn.Sequential(
#             nn.Linear(regress_hidden_dim // 2, regress_hidden_dim // 4),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(regress_hidden_dim // 4, spk_dim)
#         )
#         self.gender_clf = nn.Sequential(
#             nn.Linear(regress_hidden_dim // 2, regress_hidden_dim // 4),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(regress_hidden_dim // 4, 2)
#         )

#     def forward(self, x, lengths=None, grl_lambda=0.0):
#         x = x.permute(0, 2, 1)
#         x = self.batch_norm1(x.transpose(1, 2)).transpose(1, 2)
#         x, _ = self.lstm1(x)
        
#         # x = self.batch_norm2(x.transpose(1, 2)).transpose(1, 2)
#         # x, _ = self.lstm2(x)
#         hidden_states = self.flatten(x[:, -1, :])

#         d_emb = self.disease_proj(hidden_states)    # disease embedding
#         s_emb = self.speaker_proj(hidden_states)    # speaker embedding

#         disease_logits = self.disease_clf(d_emb)

#         # adversarial path: reverse gradients on the speaker classifier's input
#         s_in = grad_reverse(s_emb, grl_lambda) if self.training and grl_lambda > 0 else s_emb
#         speaker_logits = self.speaker_clf(s_in)
        
#         return {
#             "disease_logits": disease_logits,
#             "speaker_logits": speaker_logits,
#             "d_emb": d_emb,
#             "s_emb": s_emb
#         }

# class MyOwnCNNGradReversal(nn.Module):
#     def __init__(self, input_size, regress_hidden_dim, num_transformer_layers, output_dim, spk_dim, **kwargs):
#         super(MyOwnCNNGradReversal, self).__init__()

#         self.feature_cnn = nn.Sequential(
#             nn.Conv1d(input_size, 128, kernel_size=5, padding=2),
#             nn.ReLU(),
#             nn.BatchNorm1d(128),
#             nn.Conv1d(128, 256, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm1d(256),
#             nn.Conv1d(256, 256, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm1d(256)
#         )

#         embed_dim = 1024
#         num_heads = 16
#         dropout = 0.1

#         self.layer_norm = nn.LayerNorm(256, eps=1e-05)  # 256 is the output of feature_cnn
#         self.projection = nn.Linear(256, embed_dim)  # Project to WavLM hidden size (1024)
#         self.dropout = nn.Dropout(dropout)

#         self.pos_conv_embed = nn.Conv1d(
#             embed_dim, embed_dim, kernel_size=128, padding=64, groups=16
#         )
#         self.transformer_layer_norm = nn.LayerNorm(embed_dim, eps=1e-5)

#         self.attention_layers = nn.ModuleList([
#             nn.TransformerEncoderLayer(
#                 d_model=embed_dim,
#                 nhead=num_heads,
#                 dim_feedforward=embed_dim * 4,  # 4096 for WavLM-large
#                 dropout=dropout,
#                 activation='gelu',  # WavLM uses GELU
#                 layer_norm_eps=1e-5,
#                 batch_first=True,
#                 norm_first=True  # Pre-norm like WavLM
#             ) for _ in range(num_transformer_layers)
#         ])
#         self.final_layer_norm = nn.LayerNorm(embed_dim, eps=1e-5)
#         self.layer_weights = nn.Parameter(torch.ones(num_transformer_layers + 1))
#         self.pooling = nn.AdaptiveAvgPool1d(1)

#         # Projection layers for multi-task learning
#         self.disease_proj = nn.Sequential(
#             nn.Linear(embed_dim, embed_dim // 2),
#             nn.GELU(),  # Use GELU like WavLM
#             nn.LayerNorm(embed_dim // 2, eps=1e-5),
#             nn.Dropout(dropout)
#         )
#         self.speaker_proj = nn.Sequential(
#             nn.Linear(embed_dim, embed_dim // 2),
#             nn.GELU(),
#             nn.LayerNorm(embed_dim // 2, eps=1e-5),
#             nn.Dropout(dropout)
#         )

#         # Classifiers
#         self.disease_clf = nn.Sequential(
#             nn.Linear(embed_dim // 2, embed_dim // 4),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(embed_dim // 4, output_dim)
#         )

#         self.speaker_clf = nn.Sequential(
#             nn.Linear(embed_dim // 2, embed_dim // 4),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(embed_dim // 4, spk_dim)
#         )
        
#         self.gender_clf = nn.Sequential(
#             nn.Linear(embed_dim // 2, embed_dim // 4),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(embed_dim // 4, 2)
#         )


#     def forward(self, x, lengths=None, grl_lambda=0.0):
#         x = self.feature_cnn(x)  # torch.Size([128, 256, 32])

#         x = x.transpose(1, 2)  # [B, T, C] for layer norm
#         x = self.layer_norm(x)  # Apply layer norm
#         x = self.projection(x)   # Project to hidden size
#         x = self.dropout(x)      # Apply dropout
#         hidden_states = x        # Now has shape [B, T, hidden_size]

#         pos_conv_output = self.pos_conv_embed(hidden_states.transpose(1, 2))
#         pos_conv_output = pos_conv_output.transpose(1, 2)
#         min_seq_len = min(hidden_states.size(1), pos_conv_output.size(1))
#         hidden_states = hidden_states[:, :min_seq_len, :]
#         pos_conv_output = pos_conv_output[:, :min_seq_len, :]

#         hidden_states = hidden_states + pos_conv_output
#         hidden_states = self.transformer_layer_norm(hidden_states)

#         layer_outputs = [hidden_states]

#         for layer in self.attention_layers:
#             hidden_states = layer(hidden_states)
#             layer_outputs.append(hidden_states)
        
#         hidden_states = self.final_layer_norm(hidden_states)
#         layer_outputs[-1] = hidden_states
#         layer_weights_normalized = torch.softmax(self.layer_weights, dim=0)
#         weighted_hidden_states = torch.zeros_like(hidden_states)
#         for i, layer_output in enumerate(layer_outputs):
#             weighted_hidden_states += layer_weights_normalized[i] * layer_output
        
#         hidden_states = weighted_hidden_states.transpose(1, 2)  # [B, embed_dim, T]
#         feature_embedding = self.pooling(hidden_states).squeeze(-1)  # [B, embed_dim]
        
#         d_emb = self.disease_proj(feature_embedding)    
#         s_emb = self.speaker_proj(feature_embedding) 

#         disease_logits = self.disease_clf(d_emb)

#         s_in = grad_reverse(s_emb, grl_lambda) if self.training and grl_lambda > 0 else s_emb
#         speaker_logits = self.speaker_clf(s_in)
#         gender_logits = self.gender_clf(s_in)

#         return {
#             "disease_logits": disease_logits,
#             "speaker_logits": speaker_logits,
#             "gender_logits": gender_logits,
#             "d_emb": d_emb,
#             "s_emb": s_emb,
#         }

# class MyOwnResNet101(torchvision.models.resnet.ResNet):
#     def __init__(self, dummy, output_dim, spk_dim, track_bn=True, **kwargs):
#         def norm_layer(*args, **kwargs):
#             return nn.BatchNorm2d(*args, **kwargs, track_running_stats=track_bn)
#         super().__init__(torchvision.models.resnet.Bottleneck, [3, 4, 23, 3], norm_layer=norm_layer, num_classes=output_dim)
#         #del self.fc
#         #self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.final_feat_dim = 2048
#         self.grad_cam = False
        
#         self.preprocess = torchvision.transforms.Compose([
#             torchvision.transforms.Resize((224, 224)),
#             torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])

#         # Projection layers for multi-task learning
#         self.disease_proj = nn.Sequential(
#             nn.Linear(self.final_feat_dim, self.final_feat_dim // 2),
#             nn.GELU(),  # Use GELU like WavLM
#             nn.LayerNorm(self.final_feat_dim // 2, eps=1e-5),
#             nn.Dropout(0.1)
#         )
#         self.speaker_proj = nn.Sequential(
#             nn.Linear(self.final_feat_dim, self.final_feat_dim // 2),
#             nn.GELU(),
#             nn.LayerNorm(self.final_feat_dim // 2, eps=1e-5),
#             nn.Dropout(0.1)
#         )

#         # Classifiers
#         self.disease_clf = nn.Sequential(
#             nn.Linear(self.final_feat_dim // 2, self.final_feat_dim // 4),
#             nn.GELU(),
#             nn.Dropout(0.1),
#             nn.Linear(self.final_feat_dim // 4, output_dim)
#         )

#         self.speaker_clf = nn.Sequential(
#             nn.Linear(self.final_feat_dim // 2, self.final_feat_dim // 4),
#             nn.GELU(),
#             nn.Dropout(0.1),
#             nn.Linear(self.final_feat_dim // 4, spk_dim)
#         )
        
#         self.gender_clf = nn.Sequential(
#             nn.Linear(self.final_feat_dim // 2, self.final_feat_dim // 4),
#             nn.GELU(),
#             nn.Dropout(0.1),
#             nn.Linear(self.final_feat_dim // 4, 2)
#         )

#     def load_sl_official_weights(self, progress=True):
#         state_dict = load_state_dict_from_url(torchvision.models.resnet.ResNet101_Weights.IMAGENET1K_V2.url,
#                                               progress=progress)

#         #del state_dict['conv1.weight']
#         missing, unexpected = self.load_state_dict(state_dict, strict=False)
#         # if len(missing) > 0:
#             # raise AssertionError('Model code may be incorrect')

#     def load_ssl_official_weights(self, progress=True):
#         raise NotImplemented

#     def _forward_impl(self, x: Tensor, lengths=None) -> Tensor:
#         # See note [TorchScript super()]
#         if self.grad_cam == True:
#             x = x
#         else:
#             x = x.unsqueeze(1)
#             x = x.repeat(1, 3, 1, 1)
#             x = self.preprocess(x)

#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)

#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)

#         return x
    
#     def forward(self, x: Tensor, lengths=None, grl_lambda=0.0) -> Tensor:
#         feature_embedding = self._forward_impl(x, lengths)

#         d_emb = self.disease_proj(feature_embedding)    
#         s_emb = self.speaker_proj(feature_embedding) 

#         disease_logits = self.disease_clf(d_emb)

#         s_in = grad_reverse(s_emb, grl_lambda) if self.training and grl_lambda > 0 else s_emb
#         speaker_logits = self.speaker_clf(s_in)
#         gender_logits = self.gender_clf(s_in)

#         return {
#             "disease_logits": disease_logits,
#             "speaker_logits": speaker_logits,
#             "gender_logits": gender_logits,
#             "d_emb": d_emb,
#             "s_emb": s_emb,
#         } 

# class ResNet101(torchvision.models.resnet.ResNet):
#     def __init__(self, dummy, output_dim, track_bn=True, **kwargs):
#         def norm_layer(*args, **kwargs):
#             return nn.BatchNorm2d(*args, **kwargs, track_running_stats=track_bn)
#         super().__init__(torchvision.models.resnet.Bottleneck, [3, 4, 23, 3], norm_layer=norm_layer, num_classes=output_dim)
#         #del self.fc
#         self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.final_feat_dim = 2048
#         self.grad_cam = False
#          # TODO : Coba tambah Rezize and Normalization
#         # self.preprocess = torchvision.transforms.Compose([
#         #     torchvision.transforms.Resize((224, 224))
#         # ])

#     def load_sl_official_weights(self, progress=True):
#         state_dict = load_state_dict_from_url(torchvision.models.resnet.ResNet101_Weights.IMAGENET1K_V2.url,
#                                               progress=progress)

#         del state_dict['conv1.weight']
#         missing, unexpected = self.load_state_dict(state_dict, strict=False)
#         # if len(missing) > 0:
#             # raise AssertionError('Model code may be incorrect')

#     def load_ssl_official_weights(self, progress=True):
#         raise NotImplemented

#     def _forward_impl(self, x: Tensor, lengths=None) -> Tensor:
#         # See note [TorchScript super()]
#         if self.grad_cam == True:
#             x = x
#         else:
#             x = x.unsqueeze(1)
#             #x = self.preprocess(x)

#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)

#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)

#         return [x, 0.0, 0.0]


# class Eff_MyOwn1(nn.Module):
#     def __init__(self, input_size, regress_hidden_dim, output_dim, **kwargs):
#         super(Eff_MyOwn1, self).__init__()

#         self.target_size = (240, 240)
#         self.cnn1 = torch.nn.Conv2d(1, 3, kernel_size=3)
#         self.efficientnet = EfficientNet.from_pretrained('efficientnet-b5') #
#         # self.efficientnet = EfficientNet.from_name("efficientnet-b0", include_top=False, drop_connect_rate=0.1) # [128, 1280, 1, 1]

#         self.dense = nn.Linear(1000, 1000)
#         self.dropout = nn.Dropout(0.1)
#         self.out_proj = nn.Linear(1000, output_dim)


#     def forward(self, x):
#         x = x.permute(0, 2, 1).unsqueeze(1) # [128, 1, 94, 64]
#         x = F.interpolate(x, size=self.target_size, mode="bilinear", align_corners=False)
#         x = self.cnn1(x) # [128, 3, 92, 62]
#         x = self.efficientnet(x) # 
#         x = self.dropout(x)
#         x = self.dense(x)
#         x = torch.tanh(x)
#         x = self.dropout(x)
#         x = self.out_proj(x)
        
#         return x

# class WavLM_MyOwn1(nn.Module):
#     def __init__(self, input_size, regress_hidden_dim, output_dim, **kwargs):
#         super(WavLM_MyOwn1, self).__init__()

#         config = AutoConfig.from_pretrained("microsoft/wavlm-large")
#         self.feature_extractor = layers.WavLMFeatureEncoder(config)
#         self.feature_projection = layers.WavLMFeatureProjection(config)

#         embed_dim=1024
#         num_heads=8
#         dropout=0.1

#         self.attention = nn.TransformerEncoderLayer(
#             d_model=embed_dim,
#             nhead=num_heads,
#             dim_feedforward=embed_dim * 4,
#             dropout=dropout,
#             activation='relu',
#             batch_first=True  # [B, T, E]
#         )
#         self.encoder = nn.TransformerEncoder(self.attention, num_layers=1)
#         self.pooling = nn.AdaptiveAvgPool1d(1)  # or use mean over time dimension
        
#         #self.dse_cls = classifier_network(embed_dim, 0.1, output_dim) # CrossEntrop
#         #self.binary_classifiers = nn.ModuleList([])
#         #self.reg_classifiers = nn.ModuleList([])

#         # self.binary_classifiers.append(classifier_network(input_size, 0.1, 1150))
#         # for i in range(3):
#         #     self.binary_classifiers.append(classifier_network(input_size, 0.1,1))
#         # self.binary_classifiers.append(classifier_network(input_size, 0.1, 10))
#         self.dense = nn.Linear(embed_dim, embed_dim)
#         self.dropout = nn.Dropout(0.1)
#         self.out_proj = nn.Linear(embed_dim, output_dim)


#     def forward(self, x, lengths=None):
#         extract_features = self.feature_extractor(x)
#         extract_features = extract_features.transpose(1, 2)
#         hidden_states, extract_features = self.feature_projection(extract_features) # B, T, D

#         hidden_states = self.encoder(hidden_states)  # Self-attention over time → same shape [64, 74, 1024]
#         hidden_states = hidden_states.transpose(1, 2)  # [64, 1024, 74] for pooling over time
#         hidden_states = self.pooling(hidden_states).squeeze(-1)  # [64, 1024]
        
#         hidden_states = self.dropout(hidden_states)
#         hidden_states = self.dense(hidden_states)
#         hidden_states = torch.tanh(hidden_states)
#         hidden_states = self.dropout(hidden_states)
#         output = self.out_proj(hidden_states)
#         return output
    
# class WavLM_MyOwnGradeReversal(nn.Module):
#     def __init__(self, input_size, regress_hidden_dim, num_transformer_layers, output_dim, spk_dim, **kwargs):
#         super(WavLM_MyOwnGradeReversal, self).__init__()

#         config = AutoConfig.from_pretrained("microsoft/wavlm-large")
#         self.feature_extractor = layers.WavLMFeatureEncoder(config)
#         self.feature_projection = layers.WavLMFeatureProjection(config)

#         embed_dim = 1024
#         num_heads = 16
#         dropout = 0.1

#         self.pos_conv_embed = nn.Conv1d(
#             embed_dim, embed_dim, kernel_size=128, padding=64, groups=16
#         )
#         self.layer_norm = nn.LayerNorm(embed_dim, eps=1e-5)

#         self.attention_layers = nn.ModuleList([
#             nn.TransformerEncoderLayer(
#                 d_model=embed_dim,
#                 nhead=num_heads,
#                 dim_feedforward=embed_dim * 4,  # 4096 for WavLM-large
#                 dropout=dropout,
#                 activation='gelu',  # WavLM uses GELU
#                 layer_norm_eps=1e-5,
#                 batch_first=True,
#                 norm_first=True  # Pre-norm like WavLM
#             ) for _ in range(num_transformer_layers)
#         ])
#         self.final_layer_norm = nn.LayerNorm(embed_dim, eps=1e-5)
#         self.layer_weights = nn.Parameter(torch.ones(num_transformer_layers + 1))
#         self.pooling = nn.AdaptiveAvgPool1d(1)

#         # Projection layers for multi-task learning
#         self.disease_proj = nn.Sequential(
#             nn.Linear(embed_dim, embed_dim // 2),
#             nn.GELU(),  # Use GELU like WavLM
#             nn.LayerNorm(embed_dim // 2, eps=1e-5),
#             nn.Dropout(dropout)
#         )
#         self.speaker_proj = nn.Sequential(
#             nn.Linear(embed_dim, embed_dim // 2),
#             nn.GELU(),
#             nn.LayerNorm(embed_dim // 2, eps=1e-5),
#             nn.Dropout(dropout)
#         )

#         # Classifiers
#         self.disease_clf = nn.Sequential(
#             nn.Linear(embed_dim // 2, embed_dim // 4),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(embed_dim // 4, output_dim)
#         )

#         self.speaker_clf = nn.Sequential(
#             nn.Linear(embed_dim // 2, embed_dim // 4),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(embed_dim // 4, spk_dim)
#         )
        
#         self.gender_clf = nn.Sequential(
#             nn.Linear(embed_dim // 2, embed_dim // 4),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(embed_dim // 4, 2)
#         )


#     def forward(self, x, lengths=None, grl_lambda=0.0):
#         extract_features = self.feature_extractor(x)
#         extract_features = extract_features.transpose(1, 2)
#         hidden_states, extract_features = self.feature_projection(extract_features) # B, T, D

#         pos_conv_output = self.pos_conv_embed(hidden_states.transpose(1, 2))
#         pos_conv_output = pos_conv_output.transpose(1, 2)
#         min_seq_len = min(hidden_states.size(1), pos_conv_output.size(1))
#         hidden_states = hidden_states[:, :min_seq_len, :]
#         pos_conv_output = pos_conv_output[:, :min_seq_len, :]

#         hidden_states = hidden_states + pos_conv_output
#         hidden_states = self.layer_norm(hidden_states)

#         layer_outputs = [hidden_states]

#         for layer in self.attention_layers:
#             hidden_states = layer(hidden_states)
#             layer_outputs.append(hidden_states)
        
#         hidden_states = self.final_layer_norm(hidden_states)
#         layer_outputs[-1] = hidden_states
#         layer_weights_normalized = torch.softmax(self.layer_weights, dim=0)
#         weighted_hidden_states = torch.zeros_like(hidden_states)
#         for i, layer_output in enumerate(layer_outputs):
#             weighted_hidden_states += layer_weights_normalized[i] * layer_output
        
#         hidden_states = weighted_hidden_states.transpose(1, 2)  # [B, embed_dim, T]
#         feature_embedding = self.pooling(hidden_states).squeeze(-1)  # [B, embed_dim]
        
#         d_emb = self.disease_proj(feature_embedding)    
#         s_emb = self.speaker_proj(feature_embedding) 

#         disease_logits = self.disease_clf(d_emb)

#         s_in = grad_reverse(s_emb, grl_lambda) if self.training and grl_lambda > 0 else s_emb
#         speaker_logits = self.speaker_clf(s_in)
#         gender_logits = self.gender_clf(s_in)

#         return {
#             "disease_logits": disease_logits,
#             "speaker_logits": speaker_logits,
#             "gender_logits": gender_logits,
#             "d_emb": d_emb,
#             "s_emb": s_emb,
#         }
    
# class DownstreamMT(nn.Module):
#     def __init__(self, input_size, output_dim, **kwargs):
#         super(DownstreamMT, self).__init__()
        
#         #self.weights = nn.Parameter(torch.rand(25, 1))
#         self.pooling = nn.AdaptiveAvgPool1d(1) 

#         # Classification
#         self.dse_cls = classifier_network(input_size, 0.1, output_dim)       # CrossEntrop
#         self.binary_classifiers = nn.ModuleList([])
#         #self.reg_classifiers = nn.ModuleList([])

#         self.binary_classifiers.append(classifier_network(input_size, 0.1, 1150))
#         for i in range(3):
#             self.binary_classifiers.append(classifier_network(input_size, 0.1,1))
#         self.binary_classifiers.append(classifier_network(input_size, 0.1, 10))
#         # for i in range(2):
#         #     self.reg_classifiers.append(classifier_network(1))

#     def forward(self, x, lengths=None):
#         # layer_reps = x.hidden_states  # torch.Size([25, 8, 547, 1024])
#         # x = torch.stack(layer_reps).permute(1, 3, 2, 0)
#         # weights_normalized = nn.functional.softmax(self.weights, dim=0)
#         # feats_final = torch.matmul(x, weights_normalized.squeeze())
#         hidden_states = self.pooling(x.permute(0, 2, 1)).squeeze(-1)  # [64, 1024]

#         output = []
#         output.append(self.dse_cls(hidden_states))
#         for i, head in enumerate(self.binary_classifiers):
#             output.append(head(hidden_states))

#         # for i, head in enumerate(self.reg_classifiers):
#         #     output.append(head(hidden_states))

#         return [output, hidden_states]

# class Whisper_MyOwn1(nn.Module):
#     def __init__(self, input_size, regress_hidden_dim, output_dim, **kwargs):
#         super(Whisper_MyOwn1, self).__init__()
        
#         self.max_audio_len = 3
#         self.mel_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-large-v3", chunk_length=self.max_audio_len)
#         config = AutoConfig.from_pretrained("openai/whisper-large-v3")
#         config.max_source_positions = 150
#         self.feature_extractor = layers.WhisperFeatureEncoder(config)

#         embed_dim=1280
#         num_heads=8
#         dropout=0.1

#         self.attention = nn.TransformerEncoderLayer(
#             d_model=embed_dim,
#             nhead=num_heads,
#             dim_feedforward=embed_dim * 4,
#             dropout=dropout,
#             activation='relu',
#             batch_first=True  # [B, T, E]
#         )
#         self.encoder = nn.TransformerEncoder(self.attention, num_layers=1)
#         self.pooling = nn.AdaptiveAvgPool1d(1)  # or use mean over time dimension
#         self.classifier = nn.Linear(embed_dim, output_dim)

#     def forward(self, x, lengths=None):
#         x = self.mel_extractor(x.detach().cpu().tolist(), return_tensors="pt", sampling_rate=16000, max_length=self.max_audio_len * 16000).input_features.cuda()
#         extract_features = self.feature_extractor(x) # torch.Size([64, 150, 1280])

#         hidden_states = self.encoder(extract_features)  # Self-attention over time → same shape [64, 74, 1024]
#         hidden_states = hidden_states.transpose(1, 2)  # [64, 1024, 74] for pooling over time
#         hidden_states = self.pooling(hidden_states).squeeze(-1)  # [64, 1024]
#         out = self.classifier(hidden_states)  # [64,0 2]

#         loss_internal = 0.0
#         return [out, hidden_states, loss_internal]


# class SinusoidalPositionalEmbedding(nn.Module):
#     def __init__(self, dim, max_len=150):
#         super().__init__()
#         pe = torch.zeros(max_len, dim)
#         position = torch.arange(0, max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0)  # Shape: [1, max_len, dim]
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         # x: (batch, seq_len, dim)
#         return self.pe[:, :x.size(1)].clone().detach()

# class MyOwn2(nn.Module):
#     def __init__(self, input_size, regress_hidden_dim, output_dim, **kwargs):
#         super(MyOwn2, self).__init__()
        
#         config = AutoConfig.from_pretrained("microsoft/wavlm-large")
#         self.feature_extractor = layers.WavLMFeatureEncoder(config)
#         self.feature_projection = layers.WavLMFeatureProjection(config)

#         embed_dim=1024
#         self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
#         self.pos_embed = SinusoidalPositionalEmbedding(embed_dim) # torch.Size([1, 50, 1024])
#         self.dropout = nn.Dropout(0.1)

#         encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=1, dim_feedforward=512)
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

#         self.classifier = nn.Linear(embed_dim, output_dim)

#     def forward(self, x, lengths=None):
#         extract_features = self.feature_extractor(x)
#         extract_features = extract_features.transpose(1, 2)
#         hidden_states, extract_features = self.feature_projection(extract_features) # B, T, D

#         b, n, _ = hidden_states.shape
#         cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b) # B, 1, D
#         hidden_states = torch.cat((cls_tokens, hidden_states), dim=1) 
#         hidden_states = hidden_states + self.pos_embed(hidden_states)
#         hidden_states = self.dropout(hidden_states)

#         hidden_states = self.transformer(hidden_states)
#         cls_output = hidden_states[:, 0]
#         output = self.classifier(cls_output)
        
#         loss_internal = 0.0
#         return [output, hidden_states, loss_internal]

# class VisionTransformerCoba(torchvision.models.vision_transformer.VisionTransformer):
#     def __init__(self, dummy, output_dim, track_bn=True, **kwargs):
#         def norm_layer(*args, **kwargs):
#             return nn.LayerNorm(*args, **kwargs, eps=1e-6)
#         super().__init__(image_size=518, patch_size=14, num_layers=32, num_heads=16, hidden_dim=1280, mlp_dim=5120, norm_layer=norm_layer, num_classes=output_dim)
        
#         del self.heads
#         self.grad_cam = False

#         heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
#         heads_layers["head"] = nn.Linear(1280, output_dim)
#         self.heads = nn.Sequential(heads_layers)
#         self.preprocess = torchvision.transforms.Compose([
#             torchvision.transforms.Resize((518,518))
#         ])

#     def load_sl_official_weights(self, progress=True):
#         state_dict = load_state_dict_from_url(torchvision.models.vision_transformer.ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1.url,
#                                               progress=progress)

#         #del state_dict['conv1.weight']
#         missing, unexpected = self.load_state_dict(state_dict, strict=False)
#         # if len(missing) > 0:
#             # raise AssertionError('Model code may be incorrect')

#     def load_ssl_official_weights(self, progress=True):
#         raise NotImplemented

#     def _process_input(self, x: torch.Tensor) -> torch.Tensor:
#         n, c, h, w = x.shape
#         p = self.patch_size
#         torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
#         torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
#         n_h = h // p
#         n_w = w // p

#         # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
#         x = x.repeat(1, 3, 1, 1)
#         x = self.conv_proj(x)
#         # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
#         x = x.reshape(n, self.hidden_dim, n_h * n_w)

#         # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
#         # The self attention layer expects inputs in the format (N, S, E)
#         # where S is the source sequence length, N is the batch size, E is the
#         # embedding dimension
#         x = x.permute(0, 2, 1)

#         return x

#     def forward(self, x: torch.Tensor, lengths=None):
#         if self.grad_cam == True:
#             x = x
#         else:
#             x = x.unsqueeze(1)

#         # Reshape and permute the input tensor
#         x = self.preprocess(x)
#         x = self._process_input(x)
#         n = x.shape[0]

#         # Expand the class token to the full batch
#         batch_class_token = self.class_token.expand(n, -1, -1)
#         x = torch.cat([batch_class_token, x], dim=1)

#         x = self.encoder(x)

#         # Classifier "token" as used by standard language architectures
#         x = x[:, 0]

#         x = self.heads(x)

#         return [x, 0.0, 0.0]

# class InceptionV3(torchvision.models.inception.Inception3):
#     def __init__(self, dummy, output_dim, track_bn=True, **kwargs):
#         super().__init__(num_classes=output_dim, aux_logits=False)
#         #del self.fc
#         self.Conv2d_1a_3x3 = torchvision.models.inception.BasicConv2d(1, 32, kernel_size=3, stride=2)
#         self.final_feat_dim = 2048
#         self.preprocess = torchvision.transforms.Compose([
#             torchvision.transforms.Resize((299, 299)),
#         ])
       

#     def load_sl_official_weights(self, progress=True):
#         state_dict = load_state_dict_from_url(torchvision.models.inception.Inception_V3_Weights.IMAGENET1K_V1.url,
#                                               progress=progress)

#         del state_dict['Conv2d_1a_3x3.weight']
#         missing, unexpected = self.load_state_dict(state_dict, strict=False)
#         # if len(missing) > 0:
#             # raise AssertionError('Model code may be incorrect')

#     def load_ssl_official_weights(self, progress=True):
#         raise NotImplemented

#     def _forward(self, x: Tensor) -> Tensor:
#         x = x.unsqueeze(1)
#         x = self.preprocess(x)
#         # N x 3 x 299 x 299
#         x = self.Conv2d_1a_3x3(x)
#         # N x 32 x 149 x 149
#         x = self.Conv2d_2a_3x3(x)
#         # N x 32 x 147 x 147
#         x = self.Conv2d_2b_3x3(x)
#         # N x 64 x 147 x 147
#         x = self.maxpool1(x)
#         # N x 64 x 73 x 73
#         x = self.Conv2d_3b_1x1(x)
#         # N x 80 x 73 x 73
#         x = self.Conv2d_4a_3x3(x)
#         # N x 192 x 71 x 71
#         x = self.maxpool2(x)
#         # N x 192 x 35 x 35
#         x = self.Mixed_5b(x)
#         # N x 256 x 35 x 35
#         x = self.Mixed_5c(x)
#         # N x 288 x 35 x 35
#         x = self.Mixed_5d(x)
#         # N x 288 x 35 x 35
#         x = self.Mixed_6a(x)
#         # N x 768 x 17 x 17
#         x = self.Mixed_6b(x)
#         # N x 768 x 17 x 17
#         x = self.Mixed_6c(x)
#         # N x 768 x 17 x 17
#         x = self.Mixed_6d(x)
#         # N x 768 x 17 x 17
#         x = self.Mixed_6e(x)
#         # N x 768 x 17 x 17
#         aux: Optional[Tensor] = None
#         if self.AuxLogits is not None:
#             if self.training:
#                 aux = self.AuxLogits(x)
#         # N x 768 x 17 x 17
#         x = self.Mixed_7a(x)
#         # N x 1280 x 8 x 8
#         x = self.Mixed_7b(x)
#         # N x 2048 x 8 x 8
#         x = self.Mixed_7c(x)
#         # N x 2048 x 8 x 8
#         # Adaptive average pooling
#         x = self.avgpool(x)
#         # N x 2048 x 1 x 1
#         x = self.dropout(x)
#         # N x 2048 x 1 x 1
#         x = torch.flatten(x, 1)
#         # N x 2048
#         x = self.fc(x)
#         # N x 1000 (num_classes)
#         return x, aux
    
# class LSTMModel1(nn.Module):
#     def __init__(self, input_size, pooling_hidden, p_dropout, output_dim, **kwargs):
#         super(LSTMModel1, self).__init__()
        
#         self.batch_norm1 = nn.BatchNorm1d(input_size)
#         self.lstm1 = nn.LSTM(input_size, pooling_hidden, batch_first=True)
        
#         self.batch_norm2 = nn.BatchNorm1d(pooling_hidden)
#         self.lstm2 = nn.LSTM(pooling_hidden, pooling_hidden, batch_first=True)
        
#         #self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8, batch_first=True)

#         self.flatten = nn.Flatten()
#         self.dropout = nn.Dropout(p_dropout)
#         self.fc = nn.Linear(pooling_hidden, output_dim)

#     def forward(self, x, lengths=None):
#         # x -> [10, 13, 125]
#         x = x.permute(0, 2, 1)
#         x = self.batch_norm1(x.transpose(1, 2)).transpose(1, 2)
#         x, _ = self.lstm1(x)
        
#         x = self.batch_norm2(x.transpose(1, 2)).transpose(1, 2)
#         x, _ = self.lstm2(x)
        
#         # attn_output, _ = self.attention(x, x, x)
#         # x = torch.mean(attn_output, dim=1)

#         x = self.flatten(x[:, -1, :])
#         x = self.dropout(x)
#         x = self.fc(x)
#         return x

# class CNNClassifier(nn.Module):
#     def __init__(self, dummy, output_dim, **kwargs):
#         super(CNNClassifier, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=1)
#         self.bn1 = nn.BatchNorm2d(16)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1)
#         self.bn2 = nn.BatchNorm2d(32)
#         self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
#         self.fc = nn.Linear(32, output_dim)

#     def forward(self, x, lengths=None):
#         # x shape: [B, Feat_Dim, T]
#         x = x.unsqueeze(1)  # shape becomes [B, 1, Feat_Dim, T]
#         x = F.relu(self.bn1(self.conv1(x)))  # [B, 16, Feat_Dim, T]
#         x = F.relu(self.bn2(self.conv2(x)))  # [B, 32, Feat_Dim, T]
#         x = self.pool(x)  # [B, 32, 1, 1]
#         x = x.view(x.size(0), -1)  # Flatten to [B, 32]
#         x = self.fc(x)  # [B, num_classes]
#         return x

# ##### VIT
# class ViT(nn.Module):
#     def __init__(self, dummmy, output_dim, image_size=256, patch_size=32, dim=1024, depth=6, heads=8, mlp_dim=2048, pool = 'cls', channels = 1, dim_head = 64, dropout = 0., emb_dropout = 0., **kwargs):
#         super().__init__()
#         image_height, image_width = commons.pair(image_size)
#         patch_height, patch_width = commons.pair(patch_size)

#         assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

#         num_patches = (image_height // patch_height) * (image_width // patch_width)
#         patch_dim = channels * patch_height * patch_width
#         assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

#         self.pretrain = False
#         self.preprocess = torchvision.transforms.Compose([
#             torchvision.transforms.Resize((256,256))
#         ])

#         self.to_patch_embedding = nn.Sequential(
#             Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
#             nn.LayerNorm(patch_dim),
#             nn.Linear(patch_dim, dim),
#             nn.LayerNorm(dim),
#         )

#         self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
#         self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
#         self.dropout = nn.Dropout(emb_dropout)

#         self.transformer = modules.TransformerVIT(dim, depth, heads, dim_head, mlp_dim, dropout)

#         self.pool = pool
#         self.to_latent = nn.Identity()

#         self.mlp_head = nn.Linear(dim, output_dim)

#     def forward(self, img):
#         if self.pretrain == False:
#             img = img.unsqueeze(1)
#             img = self.preprocess(img)

#         x = self.to_patch_embedding(img)
#         b, n, _ = x.shape

#         cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
#         x = torch.cat((cls_tokens, x), dim=1)
#         x += self.pos_embedding[:, :(n + 1)]
#         x = self.dropout(x)

#         x = self.transformer(x)

#         x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

#         x = self.to_latent(x)
#         return self.mlp_head(x)

# class SimMIM(nn.Module):
#     def __init__(
#         self,
#         *,
#         encoder,
#         masking_ratio = 0.5
#     ):
#         super().__init__()
#         assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
#         self.masking_ratio = masking_ratio

#         # extract some hyperparameters and functions from encoder (vision transformer to be trained)

#         self.encoder = encoder
#         num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]

#         self.to_patch = encoder.to_patch_embedding[0]
#         self.patch_to_emb = nn.Sequential(*encoder.to_patch_embedding[1:])

#         pixel_values_per_patch = encoder.to_patch_embedding[2].weight.shape[-1]

#         # simple linear head

#         self.mask_token = nn.Parameter(torch.randn(encoder_dim))
#         self.to_pixels = nn.Linear(encoder_dim, pixel_values_per_patch)

#     def forward(self, img):
#         device = img.device

#         # get patches

#         patches = self.to_patch(img)
#         batch, num_patches, *_ = patches.shape

#         # for indexing purposes

#         batch_range = torch.arange(batch, device = device)[:, None]

#         # get positions

#         pos_emb = self.encoder.pos_embedding[:, 1:(num_patches + 1)]

#         # patch to encoder tokens and add positions

#         tokens = self.patch_to_emb(patches)
#         tokens = tokens + pos_emb

#         # prepare mask tokens

#         mask_tokens = repeat(self.mask_token, 'd -> b n d', b = batch, n = num_patches)
#         mask_tokens = mask_tokens + pos_emb

#         # calculate of patches needed to be masked, and get positions (indices) to be masked

#         num_masked = int(self.masking_ratio * num_patches)
#         masked_indices = torch.rand(batch, num_patches, device = device).topk(k = num_masked, dim = -1).indices
#         masked_bool_mask = torch.zeros((batch, num_patches), device = device).scatter_(-1, masked_indices, 1).bool()

#         # mask tokens

#         tokens = torch.where(masked_bool_mask[..., None], mask_tokens, tokens)

#         # attend with vision transformer

#         encoded = self.encoder.transformer(tokens)

#         # get the masked tokens

#         encoded_mask_tokens = encoded[batch_range, masked_indices]

#         # small linear projection for predicted pixel values

#         pred_pixel_values = self.to_pixels(encoded_mask_tokens)

#         # get the masked patches for the final reconstruction loss

#         masked_patches = patches[batch_range, masked_indices]

#         # calculate reconstruction loss

#         recon_loss = F.l1_loss(pred_pixel_values, masked_patches) / num_masked
#         return recon_loss

# class SimpleMaskedModel(nn.Module):
#     def __init__(self, dummy, output_dim, **kwargs):
#         super(SimpleMaskedModel, self).__init__()

#         self.preprocess = torchvision.transforms.Compose([
#             torchvision.transforms.Resize((256,256))
#         ])

#         self.v = ViT(
#             image_size = 256,
#             patch_size = 32,
#             num_classes = output_dim,
#             dim = 1024,
#             depth = 6,
#             heads = 8,
#             mlp_dim = 2048,
#             channels=1
#         )

#         self.mim = SimMIM(
#             encoder = self.v,
#             masking_ratio = 0.5  # they found 50% to yield the best results
#         )

#     def forward(self, x):
#         # x shape: [128, 80, 32]
#         x = x.unsqueeze(1)
#         x = self.preprocess(x)
#         loss = self.mim(x)
#         return loss

# #################### REG SSL ###########################
# class HeadCatPrediction(nn.Module):
#     def __init__(self, pooling_hidden, regress_hidden_dim, regress_dropout, regress_layers, output_dim, **kwargs):
#         super(HeadCatPrediction, self).__init__()

#         self.inp_drop = nn.Dropout(regress_dropout)
#         self.fc=nn.ModuleList([nn.Sequential(
#                 nn.Linear(pooling_hidden, regress_hidden_dim), 
#                 nn.LayerNorm(regress_hidden_dim), nn.ReLU(), nn.Dropout(regress_dropout))])

#         for lidx in range(regress_layers-1):
#             self.fc.append(nn.Sequential(
#                     nn.Linear(regress_hidden_dim, regress_hidden_dim), 
#                     nn.LayerNorm(regress_hidden_dim), nn.ReLU(), nn.Dropout(regress_dropout)))

#         self.out = nn.Sequential(nn.Linear(regress_hidden_dim, output_dim))

#         self.dense = nn.Linear(pooling_hidden, regress_hidden_dim)
#         self.dropout = nn.Dropout(regress_dropout)
#         self.out_proj = nn.Linear(regress_hidden_dim, output_dim)

#     # def get_repr(self, x):
#     #     h = self.inp_drop(x)
#     #     for lidx, fc in enumerate(self.fc):
#     #         h=fc(h)
#     #     return h

#     def forward(self, x):
#         x = self.dropout(x)
#         x = self.dense(x)
#         x = torch.tanh(x)
#         x = self.dropout(x)
#         out_dim = self.out_proj(x)
#         # h = self.get_repr(x)
#         # out_dim = self.out(h)
#         return out_dim

# import numpy as np
# class SELDModel(nn.Module):
#     """
#     SELD (Sound Event Localization and Detection) model combining ConvBlock, recurrent, and attention-based layers.
#     Supports audio-only and audio_visual input.
#     """
#     def __init__(self, params, **kwargs):
#         super().__init__()

#         self.nb_conv_blocks = 3
#         self.nb_conv_filters= 64
#         self.f_pool_size= [4, 4, 2]
#         self.t_pool_size= [5, 1, 1]
#         self.dropout= 0.05

#         self.rnn_size= 128
#         self.nb_rnn_layers= 2
#         self.nb_self_attn_layers= 2
#         self.nb_attn_heads= 8
#         self.nb_transformer_layers= 2

#         self.params = params
#         # Conv layers
#         self.conv_blocks = nn.ModuleList()
#         for conv_cnt in range(self.nb_conv_blocks):
#             self.conv_blocks.append(modules.ConvBlock(in_channels=self.nb_conv_filters if conv_cnt else 2,  # stereo
#                                               out_channels=self.nb_conv_filters,
#                                               pool_size=(self.t_pool_size[conv_cnt], self.f_pool_size[conv_cnt]),
#                                               dropout=self.dropout))

#         # GRU layers
#         self.gru_input_dim = self.nb_conv_filters * int(np.floor(80 / np.prod(self.f_pool_size)))
#         self.gru = torch.nn.GRU(input_size=self.gru_input_dim, hidden_size=self.rnn_size, num_layers=self.nb_rnn_layers,
#                                 batch_first=True, dropout=self.dropout, bidirectional=True)

#         # Self attention layers
#         self.mhsa_layers = nn.ModuleList([nn.MultiheadAttention(embed_dim=self.rnn_size, num_heads=self.nb_attn_heads,
#                                   dropout=self.dropout, batch_first=True) for _ in range(self.nb_self_attn_layers)])
#         self.layer_norms = nn.ModuleList([nn.LayerNorm(self.rnn_size) for _ in range(self.nb_self_attn_layers)])

#         # # Fusion layers
#         # if params['modality'] == 'audio_visual':
#         #     self.visual_embed_to_d_model = nn.Linear(in_features=params['resnet_feature_size'], out_features=params['rnn_size'])
#         #     self.transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=params['rnn_size'], nhead=params['nb_attn_heads'], batch_first=True)
#         #     self.transformer_decoder = nn.TransformerDecoder(self.transformer_decoder_layer, num_layers=params['nb_transformer_layers'])

#         self.fnn_list = torch.nn.ModuleList()
#         for fc_cnt in range(1):
#             self.fnn_list.append(nn.Linear(128 if fc_cnt else 128, 128, bias=True))

#         # if params['multiACCDOA']:
#         #     if params['modality'] == 'audio_visual':
#         #         self.output_dim = params['max_polyphony'] * 4 * params['nb_classes']  # 4 => (x,y), distance, on/off
#         #     else:
#         #         self.output_dim = params['max_polyphony'] * 3 * params['nb_classes']  # 3 => (x,y), distance
#         # else:
#         #     if params['modality'] == 'audio_visual':
#         #         self.output_dim = 4 * params['nb_classes']  # 4 => (x,y), distance, on/off
#         #     else:
#         #         self.output_dim = 3 * params['nb_classes']  # 3 => (x,y), distance
#         self.output_dim = 2
#         self.fnn_list.append(nn.Linear(128 if 1 else 128, self.output_dim, bias=True))

#         # self.doa_act = nn.Tanh()
#         # self.dist_act = nn.ReLU()
#         # if params['modality'] == 'audio_visual':
#         #     self.onscreen_act = nn.Sigmoid()

#     def forward(self, audio_feat, vid_feat=None):
#         """
#         Forward pass for the SELD model.
#         audio_feat: Tensor of shape (batch_size, 2, 251, 64) (stereo spectrogram input).
#         vid_feat: Optional tensor of shape (batch_size, 50, 7, 7) (visual feature map).
#         Returns:  Tensor of shape
#                   (batch_size, 50, 117) - audio - multiACCDOA.
#                   (batch_size, 50, 39)  - audio - singleACCDOA.
#                   (batch_size, 50, 156) - audio_visual - multiACCDOA.
#                   (batch_size, 50, 52) - audio_visual - singleACCDOA.

#         """
#         # audio feat - B x 2 x 251 x 64 -> 251 to 63
#         #print(audio_feat.permute(0, 2, 1).unsqueeze(1).shape)
#         audio_feat = audio_feat.permute(0, 2, 1).unsqueeze(1).repeat(1, 2, 1, 1)
#         for conv_block in self.conv_blocks:
#             audio_feat = conv_block(audio_feat)  # B x 64 x 50 x 2

#         audio_feat = audio_feat.transpose(1, 2).contiguous()  # B x 50 x 64 x 2
#         audio_feat = audio_feat.view(audio_feat.shape[0], audio_feat.shape[1], -1).contiguous()  # B x 50 x 128

#         (audio_feat, _) = self.gru(audio_feat)
#         audio_feat = torch.tanh(audio_feat)
#         audio_feat = audio_feat[:, :, audio_feat.shape[-1] // 2:] * audio_feat[:, :, :audio_feat.shape[-1] // 2]

#         for mhsa, ln in zip(self.mhsa_layers, self.layer_norms):
#             audio_feat_in = audio_feat
#             audio_feat, _ = mhsa(audio_feat_in, audio_feat_in, audio_feat_in)
#             audio_feat = audio_feat + audio_feat_in  # Residual connection
#             audio_feat = ln(audio_feat)

#         # if vid_feat is not None:
#         #     vid_feat = vid_feat.view(vid_feat.shape[0], vid_feat.shape[1], -1)  # b x 50 x 49
#         #     vid_feat = self.visual_embed_to_d_model(vid_feat)
#         #     fused_feat = self.transformer_decoder(audio_feat, vid_feat)
#         # else:
#         fused_feat = audio_feat

        
#         for fnn_cnt in range(len(self.fnn_list) - 1):
#             fused_feat = self.fnn_list[fnn_cnt](fused_feat)
#         pred = self.fnn_list[-1](fused_feat)
#         pred = pred.mean(dim=1)

#         # if self.params['modality'] == 'audio':
#         #     if self.params['multiACCDOA']:
#         #         # pred shape is batch,50,117 - 117 is 3 tracks x 3 (doa-x, doa-y, dist) x 13 classes
#         #         pred = pred.reshape(pred.size(0), pred.size(1), 3, 3, 13)
#         #         doa_pred = pred[:, :, :, 0:2, :]
#         #         dist_pred = pred[:, :, :, 2:3, :]
#         #         doa_pred = self.doa_act(doa_pred)
#         #         dist_pred = self.dist_act(dist_pred)
#         #         pred = torch.cat((doa_pred, dist_pred), dim=3)
#         #         pred = pred.reshape(pred.size(0), pred.size(1), -1)
#         #     else:
#         #         # pred shape is batch,50,39 - 39 is 3 (doa-x, doa-y, dist) x 13 classes
#         #         pred = pred.reshape(pred.size(0), pred.size(1), 3, 13)
#         #         doa_pred = pred[:, :,  0:2, :]
#         #         dist_pred = pred[:, :, 2:3, :]
#         #         doa_pred = self.doa_act(doa_pred)
#         #         dist_pred = self.dist_act(dist_pred)
#         #         pred = torch.cat((doa_pred, dist_pred), dim=2)
#         #         pred = pred.reshape(pred.size(0), pred.size(1), -1)

#         return pred
