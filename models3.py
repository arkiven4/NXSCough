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

import torchvision
from transformers import AutoConfig, AutoFeatureExtractor

from typing import Any, Callable, Optional
from collections import OrderedDict

from functools import partial

from timm.models.swin_transformer import SwinTransformerBlock
from timm.models.vision_transformer import Block

from s3prl.upstream.mockingjay.builder import PretrainedTransformer

import modules
import commons
import layers

from torch.autograd import Function


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


def grad_reverse(x, lambda_):
    return GradReverse.apply(x, lambda_)


###########################################
# OPERAGT_MAE
###########################################

class OPERAGT_MAE(nn.Module):
    def __init__(self, input_size, img_size=(256, 64), patch_size=4, mask_ratio=0.7,
                 contextual_depth=8, stride=10, temperature=.2, embed_dim=384, depth=12,
                 num_heads=6, decoder_embed_dim=256, decoder_depth=6, decoder_num_heads=8,
                 mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), norm_pix_loss=False,
                 in_chans=1, audio_exp=True, alpha=0.0, beta=4.0, mode=0, use_custom_patch=False, split_pos=False,
                 pos_trainable=False, use_nce=False, decoder_mode=1, mask_2d=False, mask_t_prob=0.5, mask_f_prob=0.5,
                 no_shift=False, output_dim=2, **kwargs):
        super(OPERAGT_MAE, self).__init__()

        # 256 -> Time, 64 mel
        self.patch_size = patch_size
        self.audio_exp = audio_exp
        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        if use_custom_patch:
            print(
                f'Use custom patch_emb with patch size: {patch_size}, stride: {stride}')
            self.patch_embed = modules.PatchEmbed_new(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, stride=stride)
        else:
            print('img_size:', img_size)
            self.patch_embed = modules.PatchEmbed_org(
                img_size, patch_size, in_chans, embed_dim)
        self.use_custom_patch = use_custom_patch
        num_patches = self.patch_embed.num_patches
        print('Using patch, patch number:', num_patches)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # self.split_pos = split_pos # not useful
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim),
            requires_grad=pos_trainable)  # fixed sin-cos embedding

        self.encoder_depth = depth
        self.contextual_depth = contextual_depth
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio,
                  qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim),
            requires_grad=pos_trainable)  # fixed sin-cos embedding
        self.no_shift = no_shift

        self.decoder_mode = decoder_mode
        if self.use_custom_patch:  # overlapped patches as in AST. Similar performance yet compute heavy
            window_size = (6, 6)
            feat_size = (102, 12)
        else:
            window_size = (4, 4)
            # feat_size = (64,8)# HxW=decoder_embed_dim
            feat_size = (32, 8)  # HxW=decoder_embed_dim

        if self.decoder_mode == 1:
            decoder_modules = []
            for index in range(16):
                if self.no_shift:
                    shift_size = (0, 0)
                else:
                    if (index % 2) == 0:
                        shift_size = (0, 0)
                    else:
                        shift_size = (2, 0)
                decoder_modules.append(
                    SwinTransformerBlock(
                        dim=decoder_embed_dim,
                        num_heads=16,
                        input_resolution=feat_size,
                        window_size=window_size,
                        shift_size=shift_size,
                        mlp_ratio=mlp_ratio,
                        proj_drop=0.0,
                        attn_drop=0.0,
                        drop_path=0.0,
                        norm_layer=norm_layer,  # nn.LayerNorm,
                    )
                )
            self.decoder_blocks = nn.ModuleList(decoder_modules)
        else:
            # Transfomer
            self.decoder_blocks = nn.ModuleList([
                Block(decoder_embed_dim, decoder_num_heads,
                      mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size**2 * in_chans, bias=True)  # decoder to patch

        # --------------------------------------------------------------------------
        self.norm_pix_loss = norm_pix_loss

        self.patch_size = patch_size
        self.stride = stride

        # audio exps
        self.alpha = alpha
        self.T = temperature
        self.mode = mode
        self.use_nce = use_nce
        self.beta = beta

        self.log_softmax = nn.LogSoftmax(dim=-1)

        self.mask_t_prob = mask_t_prob
        self.mask_f_prob = mask_f_prob
        self.mask_2d = mask_2d
        self.mask_ratio = mask_ratio

        self.initialize_weights()

        ckpt = torch.load(
            "/run/media/fourier/Data1/Pras/Thesis_Nexus/NXSCough/pretrained/encoder-operaGT.ckpt", map_location="cpu")
        self.load_state_dict(ckpt['state_dict'], strict=False)

    def initialize_weights(self):
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = modules.get_2d_sincos_pos_embed_flexible(
            self.pos_embed.shape[-1], self.patch_embed.patch_hw, cls_token=True)
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = modules.get_2d_sincos_pos_embed_flexible(
            self.decoder_pos_embed.shape[-1], self.patch_embed.patch_hw, cls_token=True)
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

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
        L = (H/p)*(W/p)
        """
        imgs = imgs.unsqueeze(1)
        p = self.patch_embed.patch_size[0]
        # assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        if self.use_custom_patch:  # overlapped patch
            h, w = self.patch_embed.patch_hw
            # todo: fixed h/w patch size and stride size. Make hw custom in the future
            x = imgs.unfold(2, self.patch_size, self.stride).unfold(
                3, self.patch_size, self.stride)  # n,1,H,W -> n,1,h,w,p,p
            x = x.reshape(shape=(imgs.shape[0], h*w, p**2 * 1))
            # x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
            # x = torch.einsum('nchpwq->nhwpqc', x)
            # x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 1))
        else:
            # print(imgs.shape)
            h = imgs.shape[2] // p
            w = imgs.shape[3] // p
            # h,w = self.patch_embed.patch_hw
            x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
            x = torch.einsum('nchpwq->nhwpqc', x)
            x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 1))

        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        specs: (N, 1, H, W)
        """
        p = self.patch_embed.patch_size[0]
        # h = 1024//p
        # w = 128//p
        h = 256//p
        # h = 64//p
        w = 64//p
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 1))
        x = torch.einsum('nhwpqc->nchpwq', x)
        specs = x.reshape(shape=(x.shape[0], 1, h * p, w * p))
        return specs

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
        # ascend: small is keep, large is remove
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def random_masking_2d(self, x, mask_t_prob, mask_f_prob):
        """
        2D: Spectrogram (msking t and f under mask_t_prob and mask_f_prob)
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        # print(N, L, D)
        if self.use_custom_patch:  # overlapped patch
            T = 101
            F = 12
        else:
            # T=64
            # F=8
            T, F = L//(64//self.patch_size), 64//self.patch_size
        # x = x.reshape(N, T, F, D)
        len_keep_t = int(T * (1 - mask_t_prob))
        len_keep_f = int(F * (1 - mask_f_prob))
        # print(T, F, len_keep_t, len_keep_f)

        # noise for mask in time
        noise_t = torch.rand(N, T, device=x.device)  # noise in [0, 1]
        # sort noise for each sample aling time
        # ascend: small is keep, large is remove
        ids_shuffle_t = torch.argsort(noise_t, dim=1)
        ids_restore_t = torch.argsort(ids_shuffle_t, dim=1)
        ids_keep_t = ids_shuffle_t[:, :len_keep_t]
        # print('t:', ids_keep_t)
        # noise mask in freq
        noise_f = torch.rand(N, F, device=x.device)  # noise in [0, 1]
        # ascend: small is keep, large is remove
        ids_shuffle_f = torch.argsort(noise_f, dim=1)
        ids_restore_f = torch.argsort(ids_shuffle_f, dim=1)
        ids_keep_f = ids_shuffle_f[:, :len_keep_f]
        # print('f:', ids_keep_f)

        # generate the binary mask: 0 is keep, 1 is remove
        # mask in freq
        mask_f = torch.ones(N, F, device=x.device)
        mask_f[:, :len_keep_f] = 0
        mask_f = torch.gather(mask_f, dim=1, index=ids_restore_f).unsqueeze(
            1).repeat(1, T, 1)  # N,T,F
        # mask in time
        mask_t = torch.ones(N, T, device=x.device)
        mask_t[:, :len_keep_t] = 0
        mask_t = torch.gather(mask_t, dim=1, index=ids_restore_t).unsqueeze(
            1).repeat(1, F, 1).permute(0, 2, 1)  # N,T,F
        mask = 1-(1-mask_t)*(1-mask_f)  # N, T, F

        # get masked x
        id2res = torch.Tensor(list(range(N*T*F))).reshape(N, T, F).to(x.device)
        # print(id2res)
        id2res = id2res  # + 999*mask # add a large value for masked elements
        id2res2 = torch.argsort(id2res.flatten(start_dim=1))
        ids_keep = id2res2.flatten(start_dim=1)[:, :len_keep_f*len_keep_t]
        # print('all ids:', ids_keep)
        assert ids_keep.max() < x.shape[1], "Index out of bounds"
        assert ids_keep.min() >= 0, "Negative index found"

        # print(x.shape)
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        # print('problem', x_masked)
        ids_restore = torch.argsort(id2res2.flatten(start_dim=1))
        mask = mask.flatten(start_dim=1)

        # print(x_masked)
        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio, mask_2d=False):
        # embed patches
        x = self.patch_embed(x)
        # print('patch embedding zise:', x.shape)
        # print('position mebedding:', self.pos_embed.shape)

        # print(self.pos_embed[:, 1:x.shape[1]+1, :].shape)

        # add pos embed w/o cls token
        pos_embed = self.pos_embed[:, 1:x.shape[1]+1, :]
        x = x + pos_embed

        # masking: length -> length * mask_ratio
        if mask_2d:
            x, mask, ids_restore = self.random_masking_2d(
                x, mask_t_prob=self.mask_t_prob, mask_f_prob=self.mask_f_prob)
            # print(x.shape)
        else:
            x, mask, ids_restore = self.random_masking(x, mask_ratio)
            # print(x.shape)
        assert torch.isfinite(self.cls_token).all(
        ), " prior cls_tokens contains NaNs or Infs"

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        # print(cls_tokens.shape)

        # Check for NaNs or Infs
        assert torch.isfinite(cls_tokens).all(
        ), "cls_tokens contains NaNs or Infs"
        assert torch.isfinite(x).all(), "x contains NaNs or Infs"

        torch.cuda.synchronize()
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        # emb = self.encoder_emb(x)

        return x, mask, ids_restore, None

    def forward_encoder_no_mask(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        # x = x + self.pos_embed[:, 1:, :]
        pos_embed = self.pos_embed[:, 1:x.shape[1]+1, :]
        x = x + pos_embed

        # masking: length -> length * mask_ratio
        # x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        contextual_embs = []
        for n, blk in enumerate(self.blocks):
            x = blk(x)
            if n > self.contextual_depth:
                contextual_embs.append(self.norm(x))
        # x = self.norm(x)
        contextual_emb = torch.stack(contextual_embs, dim=0).mean(dim=0)

        return contextual_emb

    def forward_feature(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        # x = x + self.pos_embed[:, 1:, :]
        pos_embed = self.pos_embed[:, 1:x.shape[1]+1, :]
        x = x + pos_embed

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x)
        x = x[:, 1:, :].mean(dim=1)
        x = self.norm(x)
        return x

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        # print('encoder x:', x.shape)
        x = self.decoder_embed(x)
        # print('decoder x:', x.shape)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(
            # unshuffle
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # print('append mask x:', x.shape)
        # print('position embed:', self.decoder_pos_embed.shape)
        # add pos embed

        # print(self.decoder_pos_embed.shape)
        pos_embed = self.decoder_pos_embed[:, :x.shape[1], :]
        x = x + pos_embed

        if self.decoder_mode != 0:
            B, L, D = x.shape
            x = x[:, 1:, :]
            if self.use_custom_patch:
                x = x.reshape(B, 101, 12, D)
                x = torch.cat([x, x[:, -1, :].unsqueeze(1)], dim=1)  # hack
                x = x.reshape(B, 1224, D)
        if self.decoder_mode > 3:  # mvit
            x = self.decoder_blocks(x)
        else:
            x = x.unsqueeze(1)
            for blk in self.decoder_blocks:
                x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        pred = self.decoder_pred(x)

        # remove cls token
        if self.decoder_mode != 0:
            if self.use_custom_patch:
                pred = pred.reshape(B, 102, 12, 256)
                pred = pred[:, :101, :, :]
                pred = pred.reshape(B, 1212, 256)
            else:
                pred = pred
        else:
            pred = pred[:, 1:, :]

        # pred = nn.functional.sigmoid(pred)
        return pred, None, None  # emb, emb_pixel

    def forward_loss(self, imgs, pred, mask, norm_pix_loss=False):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        # print(imgs.shape, mask.shape, mask.sum())
        target = self.patchify(imgs)
        if norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        # print(pred.shape, loss.shape)
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        # print('return:', loss)
        return loss, self.unpatchify(target), self.unpatchify(pred)

    def forward(self, imgs, attention_mask=None, **kwargs):
        # torch.optim.Adam(self.parameters(), lr=1e-4)
        imgs = imgs.permute(0, 2, 1)  # .unsqueeze(1)
        # print(self.patch_embed.proj.weight.data)
        emb_enc, mask, ids_restore, _ = self.forward_encoder(
            imgs, self.mask_ratio, mask_2d=self.mask_2d)
        pred, _, _ = self.forward_decoder(
            emb_enc, ids_restore)  # [N, L, p*p*3]
        loss_recon, target, pred = self.forward_loss(
            imgs, pred, mask, norm_pix_loss=self.norm_pix_loss)

        return {
            "loss": loss_recon,
            "pred": pred.squeeze(1),
            "imgs": imgs.squeeze(1),
        }


class Opensmile_Attention(nn.Module):
    def __init__(self, input_size, embed_dim=256, depth=4, heads=4,
                 mlp_ratio=4, output_dim=2, lstm_hidden=256, lstm_layers=2, **kwargs):
        super().__init__()

        self.feature_gate = nn.Parameter(torch.ones(input_size))

        # BiLSTM replaces CNN
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True
        )

        # Project BiLSTM output (2 * hidden) → transformer embed_dim
        self.proj = nn.Linear(lstm_hidden * 2, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=embed_dim * mlp_ratio,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=depth)

        self.pool = nn.Linear(embed_dim, 1)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, output_dim)
        )

    def forward(self, x, attention_mask=None, return_attn=False, **kwargs):
        """
        Expected x shape: [B, C=input_size, T]
        Convert to [B, T, C] for LSTM.
        """
        x = x * self.feature_gate.view(1, -1, 1)   # [B, C, T]
        x = x.transpose(1, 2)             # [B, T, C]
        h, _ = self.lstm(x)               # [B, T, 2*hidden]
        h = self.proj(h)                  # [B, T, E]

        t = self.transformer(h)
        w = torch.softmax(self.pool(t), dim=1)   # [B, T, 1]
        pooled = (w * t).sum(dim=1)              # [B, E]
        logits = self.head(pooled)

        # importance = model.feature_gate.detach().cpu()
        # norm_imp = importance / importance.sum()

        # attn = transformer_layer.self_attn.attn_output_weights   # [heads, B, T, T]
        # attn_map = attn.mean(dim=0)   # [B, T, T]
        # feature_contrib = torch.einsum("btt, bte -> bte", attn_map, hidden_states)
        # feature_importance = feature_contrib.abs().mean(dim=1)   # [B, E]

        # final_importance = (
        #     feature_gate_norm * transformer_feature_importance_norm
        # )

        return {
            "disease_logits": logits,
        }


class Eff_MyOwn1(nn.Module):
    def __init__(self, input_size, output_dim, **kwargs):
        super(Eff_MyOwn1, self).__init__()

        self.cnn1 = torch.nn.Conv2d(1, 3, kernel_size=3)
        self.efficientnet = EfficientNet.from_name(
            # [128, 1280, 1, 1]
            "efficientnet-b0", include_top=False, drop_connect_rate=0.1)

        self.dense = nn.Linear(1280, 512)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(512, output_dim)

    def forward(self, x, attention_mask=None):
        x = x.unsqueeze(1)  # [128, 1, 94, 64]
        x = self.cnn1(x)  # [128, 3, 92, 62]
        x = self.efficientnet(x).squeeze(-1).squeeze(-1)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        disease_logits = self.out_proj(x)

        return {
            "disease_logits": disease_logits,
        }


class WavLMEncoder_MyOwn(nn.Module):
    def __init__(self, input_size, num_transformer_layers=12, output_dim=2, **kwargs):
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
        self.layer_weights = nn.Parameter(
            torch.ones(num_transformer_layers + 1))
        self.pooling = nn.AdaptiveAvgPool1d(1)

        # Classifiers
        self.disease_clf = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 4, output_dim)
        )

    def forward(self, x, attention_mask=None, **kwargs):
        x = x.squeeze(1)
        extract_features = self.feature_extractor(x)
        extract_features = extract_features.transpose(1, 2)
        hidden_states, extract_features = self.feature_projection(
            extract_features)  # B, T, D

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
            weighted_hidden_states += layer_weights_normalized[i] * \
                layer_output

        hidden_states = weighted_hidden_states.transpose(
            1, 2)  # [B, embed_dim, T]
        feature_embedding = self.pooling(
            hidden_states).squeeze(-1)  # [B, embed_dim]

        disease_logits = self.disease_clf(feature_embedding)

        return {
            "embedding": feature_embedding,
            "disease_logits": disease_logits,
        }


class WavLMEncoder_MyOwnSCL(nn.Module):
    def __init__(self, input_size, num_transformer_layers=12, output_dim=2, **kwargs):
        super(WavLMEncoder_MyOwnSCL, self).__init__()

        config = AutoConfig.from_pretrained("microsoft/wavlm-large")
        self.feature_extractor = layers.WavLMFeatureEncoder(config)
        self.feature_projection = layers.WavLMFeatureProjection(config)

        embed_dim = 1024
        num_heads = 16
        dropout = 0.1
        self.temperature = 0.1

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
        self.layer_weights = nn.Parameter(
            torch.ones(num_transformer_layers + 1))

        # self.pooling = nn.AdaptiveAvgPool1d(1)
        self.pooling = layers.AttentiveStatisticsPooling(embed_dim)
        embed_dim = embed_dim * 2

        self.projector = layers.CustomWalvmProjector(
            dim_in=embed_dim,
            dim_hidden=4096,
            dim_out=256
        )

    def calc_cl(self, z1, z2):
        """
        z1, z2: [batch, dim] embeddings from two augmented views
        """
        B = z1.size(0)

        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        logits = torch.matmul(z1, z2.t()) / self.temperature
        labels = torch.arange(B, device=z1.device)

        loss_12 = F.cross_entropy(logits, labels)
        loss_21 = F.cross_entropy(logits.t(), labels)

        return (loss_12 + loss_21) * 0.5

    def forward_encoder(self, x, attention_mask=None):
        x = x.squeeze(1)
        extract_features = self.feature_extractor(x)
        extract_features = extract_features.transpose(1, 2)
        hidden_states, extract_features = self.feature_projection(
            extract_features)  # B, T, D

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
            weighted_hidden_states += layer_weights_normalized[i] * \
                layer_output

        # hidden_states = weighted_hidden_states.transpose(1, 2)  # [B, embed_dim, T]
        # .transpose(1, 2)  # [B, embed_dim, T]
        hidden_states = weighted_hidden_states

        feature_embedding = self.pooling(
            hidden_states).squeeze(-1)  # [B, embed_dim]
        z = self.projector(feature_embedding)
        return feature_embedding, z

    def forward(self, x, x2=None, attention_mask=None):
        _, z1 = self.forward_encoder(x)
        _, z2 = self.forward_encoder(x2)

        return {
            "loss": self.calc_cl(z1, z2),
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

        # self.pooling = layers.AttentiveStatisticsPooling(input_size, attention_dim=128)
        # embed_dim = input_size * 2  # Since ASP outputs 2*input_size
        # num_clusters = 4
        # embed_dim = input_size * num_clusters
        # self.pooling = layers.NetVLAD(num_clusters=num_clusters, dim=input_size, alpha=1.0)

        # self.proj = nn.Sequential(
        #     nn.LayerNorm(embed_dim),
        #     nn.Linear(embed_dim, 1024),
        #     nn.GELU()
        # )
        # self.classifier = nn.Linear(1024, output_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, 1)
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

        # feature_embedding = self.pooling(features, length)
        # feature_embedding = self.pooling(features.permute(0, 2, 1).unsqueeze(-1))
        # feature_embedding = self.proj(feature_embedding)
        feature_embedding = features.mean(dim=1)
        disease_logits = self.classifier(feature_embedding)

        return {
            "disease_logits": disease_logits,
            "embedding": feature_embedding,
        }


class PEFTWavLM_SCCL(nn.Module):
    def __init__(self, input_size, output_dim, lora_rank, lora_alpha, target_modules, spk_dim, lora_finetune, **kwargs):
        super(PEFTWavLM_SCCL, self).__init__()

        from transformers import WavLMModel
        from peft import get_peft_model, LoraConfig, TaskType

        self.backbone_model = WavLMModel.from_pretrained(
            "microsoft/wavlm-large",
            output_hidden_states=True
        )
        self.backbone_model.freeze_feature_encoder()

        self.pooling = layers.AttentiveStatisticsPooling(
            input_size, attention_dim=128)
        embed_dim = input_size * 2  # Since ASP outputs 2*input_size

        temp_ckpt = torch.load("logs/peftwavlm_ssccl_patient/best_model.ckpt")
        raw_sd = temp_ckpt["state_dict"]
        patched_sd = {}
        for k, v in raw_sd.items():
            patched_key = k[len("model."):] if k.startswith("model.") else k
            patched_sd[patched_key] = v
        # True for Test and First
        self.load_state_dict(patched_sd, strict=False)

        if lora_finetune:
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=0.1,
                target_modules=target_modules,
            )
            self.backbone_model = get_peft_model(
                self.backbone_model, lora_config)

        # self.projector = nn.Sequential(
        #     nn.Linear(embed_dim, embed_dim),
        #     nn.ReLU(),
        #     nn.Linear(embed_dim, embed_dim),
        # )
        self.proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 128),
            nn.GELU()
        )
        self.classifier = nn.Linear(128, output_dim)

    def disabled_calc_additional_loss(self, outputs, batch):
        # ---------------------- Phase 1 --------------------------
        # """
        # z1, z2: (B, D)
        # patient_ids: list[str] or tensor (B,)
        # """
        # _, _, _, _, dse_ids, [patient_ids, _] = batch
        # z1 = outputs['z1']
        # z2 = outputs['z2']

        # temperature=0.07
        # device = z1.device
        # B = z1.size(0)

        # z = torch.cat([z1, z2], dim=0)        # (2B, D)
        # z = F.normalize(z, dim=1)

        # patient_ids = torch.tensor(
        #     [hash(p) for p in patient_ids],
        #     device=device
        # )
        # patient_ids = patient_ids.repeat(2)   # (2B,)

        # sim = torch.matmul(z, z.T) / temperature

        # mask = torch.eye(2 * B, device=device).bool()
        # sim.masked_fill_(mask, -1e9)

        # pos_mask = patient_ids.unsqueeze(0) == patient_ids.unsqueeze(1)
        # pos_mask = pos_mask & (~mask)

        # exp_sim = torch.exp(sim)
        # log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))

        # mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / pos_mask.sum(dim=1).clamp(min=1)
        # loss = -mean_log_prob_pos.mean()
        # return loss
        return 0

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

    def _forward_encoder(self, x, attention_mask=None, **kwargs):
        x = x.squeeze(1)
        with torch.no_grad():
            x = self.backbone_model.feature_extractor(x)
            x = x.transpose(1, 2)  # New version of huggingface
            x, _ = self.backbone_model.feature_projection(
                x)  # New version of huggingface

        length = None
        if attention_mask is not None:
            length = commons.compute_length_from_mask(
                attention_mask.detach().cpu())
            length = torch.tensor(length).cuda()

        x = self.backbone_model.encoder(
            x, output_hidden_states=True
        )
        features = x.last_hidden_state  # torch.Size([32, 24, 1024])

        feature_embedding = self.pooling(features, length)
        # feature_embedding = self.projector(feature_embedding)
        return feature_embedding

    def forward(self, x, x2, attention_mask=None, **kwargs):
        z1 = self._forward_encoder(x, attention_mask=None)
        # z2 = self._forward_encoder(x2, attention_mask=None)
        feature_embedding = self.proj(z1)
        disease_logits = self.classifier(feature_embedding)

        return {
            "disease_logits": disease_logits,
            # "z1": z1,
            # "z2": z2
        }


class PEFTWavlm_CoughDetection(nn.Module):
    def __init__(self, input_size, output_dim, spk_dim, **kwargs):
        super(PEFTWavlm_CoughDetection, self).__init__()

        self.pooling = layers.AttentiveStatisticsPooling(
            input_size, attention_dim=128)
        embed_dim = input_size * 2  # Since ASP outputs 2*input_size
        dropout = 0.1

        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(embed_dim // 2, 1)

    def forward(self, x, attention_mask=None, **kwargs):
        feature_embedding = self.pooling(x)

        x_proj = self.proj(feature_embedding)
        logits = self.classifier(x_proj).squeeze(-1)

        return {
            "disease_logits": logits,
            "embedding": feature_embedding,
        }


class DownstreamWavLMEncoder_MyOwnSCL(nn.Module):
    def __init__(
            self, input_size, num_transformer_layers=12, output_dim=2, freeze_encoder=False, use_proj_output=True, **
            kwargs):
        super(DownstreamWavLMEncoder_MyOwnSCL, self).__init__()

        self.use_proj_output = use_proj_output
        self.sscl_model = WavLMEncoder_MyOwnSCL(1000, 12, 2)
        temp_ckpt = torch.load(
            "/run/media/fourier/Data1/Pras/Thesis_Nexus/NXSCough/logs/wavlmencoder_scl_attentive/fold_0/pool_fold0_epoch=03.ckpt")
        raw_sd = temp_ckpt["state_dict"]
        patched_sd = {}
        for k, v in raw_sd.items():
            patched_key = k[len("model."):] if k.startswith("model.") else k
            patched_sd[patched_key] = v
        self.sscl_model.load_state_dict(patched_sd, strict=False)
        self.sscl_model.feature_extractor._freeze_parameters()

        if freeze_encoder:
            for p in self.sscl_model.parameters():
                p.requires_grad = False

        in_dim = 256 if use_proj_output else 2048
        self.head = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(in_dim // 2, output_dim)
        )

    def forward(self, x, x2=None, attention_mask=None):
        out = self.sscl_model.forward_encoder(x)
        if self.use_proj_output:
            feature_embedding = out[1]
        else:
            feature_embedding = out[0]

        disease_logits = self.head(feature_embedding)
        return {
            "embedding": feature_embedding,
            "disease_logits": disease_logits,
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

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.audiotower_hidden_dim),
            nn.Linear(self.audiotower_hidden_dim, 1)
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
        # audio_features = pad_sequence(audio_features, batch_first=True) # for Attentive Pooling
        audio_features = torch.stack([x.mean(dim=0)
                                     for x in audio_features], dim=0)
        audio_features = audio_features.to(torch.float32)

        disease_logits = self.classifier(audio_features)

        return {
            "disease_logits": disease_logits,
        }


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

        pretrained_dict = torch.load(
            "pretrained/tera_pretrained.pth", weights_only=False)
        transformer_state = pretrained_dict['Transformer']

        self.tera_model = PretrainedTransformer(options, inp_dim=-1)
        self.tera_model.model.load_state_dict(transformer_state, strict=True)
        for param in self.tera_model.parameters():
            param.requires_grad = False
        self.tera_model.eval()

        self.pooling = layers.AttentiveStatisticsPooling(
            self.tera_model.model_config.hidden_size, attention_dim=128)
        embed_dim = self.tera_model.model_config.hidden_size * 2

        self.projector = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.LayerNorm(256),
            nn.GELU()
        )

        self.tb_head = nn.Linear(256, 1)

    def calc_additional_loss(self, outputs, batch):
        """
        z1, z2: (B, D)
        patient_ids: list[str] or tensor (B,)
        """
        beta = 0.0
        _, _, _, _, dse_ids, [_, _] = batch
        if len(dse_ids.shape) == 2:
            dse_ids = torch.argmax(dse_ids, dim=1)

        labels = dse_ids
        logits = outputs['disease_logits']

        pi = 0.3  # 0.5 * (labels.mean().item())

        def loss_fn(x):
            return F.binary_cross_entropy_with_logits(
                x, torch.ones_like(x), reduction="none"
            )

        def loss_fn_neg(x):
            return F.binary_cross_entropy_with_logits(
                x, torch.zeros_like(x), reduction="none"
            )

        pos = logits[labels == 1]   # unlabeled
        neg = logits[labels == 0]   # clean negatives

        if len(neg) == 0:
            return torch.tensor(0.0, device=logits.device)

        # Risk estimators
        risk_neg = loss_fn_neg(neg).mean()
        risk_pos = loss_fn(pos).mean() if len(pos) > 0 else 0.0
        risk_unl = loss_fn_neg(pos).mean() if len(pos) > 0 else 0.0

        # PU risk
        risk = pi * risk_pos + risk_unl - pi * risk_neg

        # Non-negative correction
        return torch.clamp(risk, min=beta)

    def forward(self, x, attention_mask=None, **kwargs):
        x = x.squeeze(1)
        # with torch.no_grad():
        # Index 0 = Last Hidden, Index 1 All Transformwer
        x = self.tera_model(x)[0]
        x = torch.nan_to_num(x, nan=0.0)

        feature_embedding = self.pooling(x)
        feature_embedding = self.projector(feature_embedding)

        disease_logits = self.tb_head(feature_embedding).squeeze(-1)
        return {
            "disease_logits": disease_logits,
            "embedding": feature_embedding,
        }


class ResNet152Classifier(nn.Module):
    def __init__(self, dummy_input, output_dim=2, pretrained=True, **kwargs):
        super().__init__()

        # Backbone provisioning
        from torchvision import models
        self.backbone = models.resnet152(
            weights=models.ResNet152_Weights.DEFAULT if pretrained else None)
        self.backbone.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=self.backbone.conv1.out_channels,
            kernel_size=self.backbone.conv1.kernel_size,
            stride=self.backbone.conv1.stride,
            padding=self.backbone.conv1.padding,
            bias=self.backbone.conv1.bias
        )

        # Replace final FC for downstream task alignment
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, output_dim)

    def forward(self, x, attention_mask=None, **kwargs):
        x = x.unsqueeze(1)
        disease_logits = self.backbone(x)
        return {
            "disease_logits": disease_logits,
        }

# ResNet(BasicBlock, [3, 4, 6, 3],
#                   feat_dim=feat_dim,
#                   embed_dim=embed_dim,
#                   pooling_func=pooling_func,
#                   two_emb_layer=two_emb_layer)

class SharedRes2NetTripleASP(nn.Module):
    def __init__(self, dummy, output_dim=2, **kwargs):
        super().__init__()

        self.encoder = Res2NetAudio(base_channels=512, scale=4)

        self.fusion_logits = nn.Parameter(torch.zeros(3))
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_dim)
        )

    def forward(self, x, **kwargs):
        # x: (B, 240, T)
        streams = torch.split(x, 80, dim=1)

        feats = []
        for s in streams:
            z = self.encoder(s)   # (B, 64, T')
            z = z.mean(dim=-1)    # (B, 128)
            feats.append(z)

        feats = torch.stack(feats, dim=1)      # (B, 3, 128)

        w = torch.softmax(self.fusion_logits, dim=0)
        fused = torch.sum(feats * w[None, :, None], dim=1)

        disease_logits = self.classifier(fused)
        return {
            "disease_logits": disease_logits,
        }


class ResNet34ManualClassifier(nn.Module):
    def __init__(
        self,
        dummy_input,
        feature_dim: int = 39,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_tabular: bool = False,
        output_dim: int = 2, **kwargs
    ):
        super().__init__()

        block = getattr(modules, "BasicBlock")
        num_blocks = [3, 4, 6, 3]
        m_channels = 32

        self.in_planes = m_channels
        self.feature_dim = feature_dim
        # Compute frequency downsampling after 3 stride-2 stages precisely:
        # H_next = floor((H - 1) / 2) + 1 for k=3, p=1, s=2
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
            nn.Dropout(dropout),
            nn.Linear(128, output_dim)
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
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        stats = self.pool(out)
        disease_logits = self.classifier(stats)

        return {
            "disease_logits": disease_logits,
        }

class TemporalAttention1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.query_conv = nn.Conv1d(channels, channels // 8, kernel_size=1)
        self.key_conv   = nn.Conv1d(channels, channels // 8, kernel_size=1)
        self.value_conv = nn.Conv1d(channels, channels, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        x: (B, C, T)
        """
        B, C, T = x.size()

        query = self.query_conv(x).permute(0, 2, 1)   # B, T, C'
        key   = self.key_conv(x)                      # B, C', T
        energy = torch.bmm(query, key)                # B, T, T
        attention = self.softmax(energy)

        value = self.value_conv(x)                    # B, C, T
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(B, C, T)

        return self.gamma * out + x
    
class ChannelAttention1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        x: (B, C, T)
        """
        B, C, T = x.size()

        proj_query = x.view(B, C, -1)          # B, C, T
        proj_key   = x.view(B, C, -1).permute(0, 2, 1)  # B, T, C
        energy = torch.bmm(proj_query, proj_key)        # B, C, C

        energy_new = torch.max(energy, dim=-1, keepdim=True)[0] - energy
        attention = self.softmax(energy_new)

        out = torch.bmm(attention, proj_query)  # B, C, T
        return self.gamma * out + x

class ResNet34AttClassifier(nn.Module):
    def __init__(
        self,
        dummy_input,
        feature_dim: int = 39,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_tabular: bool = False,
        output_dim: int = 2, **kwargs
    ):
        super().__init__()

        block = getattr(modules, "BasicBlock")
        num_blocks = [3, 4, 6, 3]
        m_channels = 32

        self.in_planes = m_channels
        self.feature_dim = feature_dim
        # Compute frequency downsampling after 3 stride-2 stages precisely:
        # H_next = floor((H - 1) / 2) + 1 for k=3, p=1, s=2
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

        self.tam = TemporalAttention1D(256)
        self.cam = ChannelAttention1D()

        self.pool = nn.AdaptiveAvgPool1d(1)
        # self.pool = modules.TSTP(in_dim=self.stats_dim * block.expansion)
        # self.pool_out_dim = self.pool.get_out_dim()

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim)
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
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = out.squeeze(-2)
        out = self.tam(out) + self.cam(out)
        stats = self.pool(out).squeeze(-1)
        
        disease_logits = self.classifier(stats)

        return {
            "disease_logits": disease_logits,
        }


class BiLSTMMelClassifier(nn.Module):
    def __init__(
        self,
        dummy_input,
        feature_dim: int = 39,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_tabular: bool = False,
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

        self.audio_project = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
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
                nn.Dropout(0.5),          # strong regularization (intentional)
                nn.Linear(32, 64),
                nn.ReLU(),
            )

            # Project tabular → audio space
            self.tabular_project = nn.Linear(64, 256)

            # Gate: decides how much tabular matters
            self.gate = nn.Sequential(
                nn.Linear(64, 256),
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
        audio_feat = audio_feat.mean(dim=1)           # temporal pooling
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


class BiLSTMSelfAttClassifier(nn.Module):
    def __init__(
        self,
        dummy,
        feature_dim=39,
        hidden_size=256,
        num_layers=2,
        dropout=0.1,
        num_heads=4,
        output_dim=2,
        **kwargs
    ):
        super().__init__()

        # -------------------------
        # 1. BiLSTM backbone
        # -------------------------
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        lstm_dim = hidden_size * 2

        # -------------------------
        # 2. Self-attention block (stable: projection + LayerNorm)
        # -------------------------
        self.attn_norm = nn.LayerNorm(lstm_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=lstm_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.attn_proj = nn.Sequential(
            nn.Linear(lstm_dim, lstm_dim),
            nn.Dropout(dropout)
        )

        # -------------------------
        # 3. Statistics-attentive pooling
        #    Outputs: [weighted_mean || weighted_std]
        # -------------------------
        self.pool_attn = nn.Sequential(
            nn.Linear(lstm_dim, lstm_dim // 4),
            nn.Tanh(),
            nn.Linear(lstm_dim // 4, 1)
        )
        pooled_dim = lstm_dim * 2  # mean + std

        # -------------------------
        # 4. Classification head
        # -------------------------
        self.project = nn.Sequential(
            nn.Linear(pooled_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_dim)
        )

    def attentive_stats_pooling(self, x):
        # x: (B, T, C)
        attn_weights = torch.softmax(self.pool_attn(x), dim=1)  # (B, T, 1)
        mean = torch.sum(attn_weights * x, dim=1)
        var = torch.sum(attn_weights * (x - mean.unsqueeze(1)) ** 2, dim=1)
        std = torch.sqrt(var + 1e-6)
        return torch.cat([mean, std], dim=-1)

    def forward(self, x, **kwargs):
        x = x.permute(0, 2, 1)          # (B, T, F)

        # BiLSTM
        out, _ = self.lstm(x)           # (B, T, 2H)

        # Self-attention block
        normed = self.attn_norm(out)
        attn_out, _ = self.self_attn(normed, normed, normed)
        out = out + self.attn_proj(attn_out)  # residual

        # Statistics-attentive pooling
        pooled = self.attentive_stats_pooling(out)

        # Classifier
        logits = self.project(pooled)

        return {"disease_logits": logits}

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

    def forward(self, x, attention_mask=None, **kwargs):
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
    

class AttentiveStatsPooling(nn.Module):
    def __init__(self, in_dim, bottleneck_dim=128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(in_dim, bottleneck_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(bottleneck_dim, in_dim, kernel_size=1),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        # x: (B, C, T)
        alpha = self.attention(x)
        mean = torch.sum(alpha * x, dim=-1)
        var = torch.sum(alpha * (x - mean.unsqueeze(-1)) ** 2, dim=-1)
        std = torch.sqrt(var + 1e-9)
        return torch.cat([mean, std], dim=1)

class Res2NetBlock(nn.Module):
    def __init__(self, channels, scale=4):
        super().__init__()
        assert channels % scale == 0
        self.scale = scale
        self.width = channels // scale

        self.convs = nn.ModuleList([
            nn.Conv2d(self.width, self.width, kernel_size=3, padding=1, bias=False)
            for _ in range(scale - 1)
        ])
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        splits = torch.split(x, self.width, dim=1)
        out = [splits[0]]

        for i in range(1, self.scale):
            if i == 1:
                y = self.convs[i - 1](splits[i])
            else:
                y = self.convs[i - 1](splits[i] + out[i - 1])
            out.append(y)

        out = torch.cat(out, dim=1)
        out = self.bn(out)
        return self.relu(out + x)

class Res2NetAudio(nn.Module):
    def __init__(self, base_channels=64, scale=4):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        self.layer1 = Res2NetBlock(base_channels, scale)
        self.layer2 = Res2NetBlock(base_channels, scale)
        self.layer3 = Res2NetBlock(base_channels, scale)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.stem(x)                 # (B, C, F', T')
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = torch.mean(x, dim=2)         # freq pooling → (B, C, T')
        return x

class AdaptiveConvPooling(nn.Module):
    def __init__(self, in_dim, out_dim=256):
        super().__init__()
        self.pool = nn.Sequential(
            nn.Conv1d(in_dim, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):
        # x: (B, C, T)
        x = self.pool(x)
        return x.squeeze(-1)  # (B, out_dim)

class SharedRes2NetTripleASP(nn.Module):
    def __init__(self, dummy, output_dim=2, **kwargs):
        super().__init__()

        self.encoder = Res2NetAudio(base_channels=512, scale=4)

        self.fusion_logits = nn.Parameter(torch.zeros(3))
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_dim)
        )

    def forward(self, x, **kwargs):
        # x: (B, 240, T)
        streams = torch.split(x, 80, dim=1)

        feats = []
        for s in streams:
            z = self.encoder(s)   # (B, 64, T')
            z = z.mean(dim=-1)    # (B, 128)
            feats.append(z)

        feats = torch.stack(feats, dim=1)      # (B, 3, 128)

        w = torch.softmax(self.fusion_logits, dim=0)
        fused = torch.sum(feats * w[None, :, None], dim=1)

        disease_logits = self.classifier(fused)
        return {
            "disease_logits": disease_logits,
        }
