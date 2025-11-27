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

import modules
import commons
import layers

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
            print(f'Use custom patch_emb with patch size: {patch_size}, stride: {stride}')
            self.patch_embed = modules.PatchEmbed_new(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, stride=stride)
        else:
            print('img_size:', img_size)
            self.patch_embed = modules.PatchEmbed_org(img_size, patch_size, in_chans, embed_dim)
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
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
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
                Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True)  # decoder to patch

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

    def initialize_weights(self):
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = modules.get_2d_sincos_pos_embed_flexible(
            self.pos_embed.shape[-1], self.patch_embed.patch_hw, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = modules.get_2d_sincos_pos_embed_flexible(
            self.decoder_pos_embed.shape[-1], self.patch_embed.patch_hw, cls_token=True)
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
        ids_shuffle_t = torch.argsort(noise_t, dim=1)  # ascend: small is keep, large is remove
        ids_restore_t = torch.argsort(ids_shuffle_t, dim=1)
        ids_keep_t = ids_shuffle_t[:, :len_keep_t]
        # print('t:', ids_keep_t)
        # noise mask in freq
        noise_f = torch.rand(N, F, device=x.device)  # noise in [0, 1]
        ids_shuffle_f = torch.argsort(noise_f, dim=1)  # ascend: small is keep, large is remove
        ids_restore_f = torch.argsort(ids_shuffle_f, dim=1)
        ids_keep_f = ids_shuffle_f[:, :len_keep_f]
        # print('f:', ids_keep_f)

        # generate the binary mask: 0 is keep, 1 is remove
        # mask in freq
        mask_f = torch.ones(N, F, device=x.device)
        mask_f[:, :len_keep_f] = 0
        mask_f = torch.gather(mask_f, dim=1, index=ids_restore_f).unsqueeze(1).repeat(1, T, 1)  # N,T,F
        # mask in time
        mask_t = torch.ones(N, T, device=x.device)
        mask_t[:, :len_keep_t] = 0
        mask_t = torch.gather(mask_t, dim=1, index=ids_restore_t).unsqueeze(1).repeat(1, F, 1).permute(0, 2, 1)  # N,T,F
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
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
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
            x, mask, ids_restore = self.random_masking_2d(x, mask_t_prob=self.mask_t_prob, mask_f_prob=self.mask_f_prob)
            # print(x.shape)
        else:
            x, mask, ids_restore = self.random_masking(x, mask_ratio)
            # print(x.shape)
        # print('x_mask:', x.shape)
        # append cls token
        # print('here')
        # print(self.pos_embed)
        # print(self.cls_token)
        assert torch.isfinite(self.cls_token).all(), " prior cls_tokens contains NaNs or Infs"

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        # print(cls_tokens.shape)

        # Check for NaNs or Infs
        assert torch.isfinite(cls_tokens).all(), "cls_tokens contains NaNs or Infs"
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
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
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

    def forward(self, imgs, attention_mask=None):
        # torch.optim.Adam(self.parameters(), lr=1e-4)
        imgs = imgs.permute(0, 2, 1)  # .unsqueeze(1)
        # print(self.patch_embed.proj.weight.data)
        emb_enc, mask, ids_restore, _ = self.forward_encoder(imgs, self.mask_ratio, mask_2d=self.mask_2d)
        pred, _, _ = self.forward_decoder(emb_enc, ids_restore)  # [N, L, p*p*3]
        loss_recon, target, pred = self.forward_loss(imgs, pred, mask, norm_pix_loss=self.norm_pix_loss)

        return {
            "loss": loss_recon,
            "pred": pred.squeeze(1),
            "imgs": imgs.squeeze(1),
        }


class Opensmile_Attention(nn.Module):
    def __init__(self, input_size, cnn_channels=128, embed_dim=128, num_heads=4, output_dim=2, **kwargs):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(cnn_channels, embed_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, output_dim)
        )

    def forward(self, x, attention_mask=None, return_attn=False, **kwargs):
        # x: [B, 376, T] where T is dynamic

        h = self.cnn(x)                # [B, embed_dim, T]
        h = h.transpose(1, 2)          # [B, T, embed_dim]
        attn_out, attn_weights = self.attn(h, h, h)
        pooled = attn_out.mean(dim=1)  # [B, embed_dim]
        logits = self.head(pooled)     # [B, 2]

        if return_attn:
            return {
                "disease_logits": logits,
                "attn_weights": attn_weights,
            }
        
        return {
            "disease_logits": logits,
        }


class Eff_MyOwn1(nn.Module):
    def __init__(self, input_size, output_dim, **kwargs):
        super(Eff_MyOwn1, self).__init__()

        self.cnn1 = torch.nn.Conv2d(1, 3, kernel_size=3)
        self.efficientnet = EfficientNet.from_name(
            "efficientnet-b0", include_top=False, drop_connect_rate=0.1)  # [128, 1280, 1, 1]

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
