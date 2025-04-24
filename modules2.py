import os
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.container import ModuleList
from fairseq.modules import SinusoidalPositionalEmbedding
from abc import ABC


def clean_state_dict(state_dict):
    new = {}
    for key, value in state_dict.items():
        if key in ['fc.weight', 'fc.bias']:
            continue
        new[key.replace('bert.', '')] = value
    return new
    
def _get_clones(module, n):
    return ModuleList([copy.deepcopy(module) for _ in range(n)])


class TransformerBlock(nn.Module, ABC):
    def __init__(self,
                 d_model,
                 n_heads,
                 attn_dropout,
                 res_dropout):
        super(TransformerBlock, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=attn_dropout)
        self.dropout = nn.Dropout(res_dropout)

    def forward(self,
                query, key, value,
                key_padding_mask=None,
                attn_mask=True):
        """
        From original Multimodal Transformer code,

        In the original paper each operation (multi-head attention or FFN) is
        post-processed with: `dropout -> add residual -> layer-norm`. In the
        tensor2tensor code they suggest that learning is more robust when
        preprocessing each layer with layer-norm and postprocessing with:
        `dropout -> add residual`. We default to the approach in the paper.
        """
        query, key, value = [self.layer_norm(x) for x in (query, key, value)]
        mask = self.get_future_mask(query, key) if attn_mask else None
        x = self.self_attn(
            query, key, value,
            key_padding_mask=key_padding_mask,
            attn_mask=mask)[0]
        return query + self.dropout(x)

    @staticmethod
    def get_future_mask(query, key=None):
        """
        :return: source mask
            ex) tensor([[0., -inf, -inf],
                        [0., 0., -inf],
                        [0., 0., 0.]])
        """
        dim_query = query.shape[0]
        dim_key = dim_query if key is None else key.shape[0]

        future_mask = torch.ones(dim_query, dim_key, device=query.device)
        future_mask = torch.triu(future_mask, diagonal=1).float()
        future_mask = future_mask.masked_fill(future_mask == float(1), float('-inf'))
        return future_mask


class FeedForwardBlock(nn.Module, ABC):
    def __init__(self,
                 d_model,
                 d_feedforward,
                 res_dropout,
                 relu_dropout):
        super(FeedForwardBlock, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_feedforward)
        self.dropout1 = nn.Dropout(relu_dropout)
        self.linear2 = nn.Linear(d_feedforward, d_model)
        self.dropout2 = nn.Dropout(res_dropout)

    def forward(self, x):
        """
        Do layer-norm before self-attention
        """
        normed = self.layer_norm(x)
        projected = self.linear2(self.dropout1(F.relu(self.linear1(normed))))
        skipped = normed + self.dropout2(projected)
        return skipped


class TransformerEncoderBlock(nn.Module, ABC):
    def __init__(self,
                 d_model,
                 n_heads,
                 d_feedforward,
                 attn_dropout,
                 res_dropout,
                 relu_dropout):
        """
        Args:
            d_model: the number of expected features in the input (required).
            n_heads: the number of heads in the multi-head attention models (required).
            d_feedforward: the dimension of the feedforward network model (required).
            attn_dropout: the dropout value for multi-head attention (required).
            res_dropout: the dropout value for residual connection (required).
            relu_dropout: the dropout value for relu (required).
        """
        super(TransformerEncoderBlock, self).__init__()
        self.transformer = TransformerBlock(d_model, n_heads, attn_dropout, res_dropout)
        self.feedforward = FeedForwardBlock(d_model, d_feedforward, res_dropout, relu_dropout)

    def forward(self,
                x_query,
                x_key=None,
                key_mask=None,
                attn_mask=None):
        """
        x : input of the encoder layer -> (L, B, d)
        """
        if x_key is not None:
            x = self.transformer(
                x_query, x_key, x_key,
                key_padding_mask=key_mask,
                attn_mask=attn_mask
            )
        else:
            x = self.transformer(
                x_query, x_query, x_query,
                key_padding_mask=key_mask,
                attn_mask=attn_mask
            )
        x = self.feedforward(x)
        return x


class CrossmodalTransformer(nn.Module, ABC):
    def __init__(self,
                 n_layers,
                 n_heads,
                 d_model,
                 attn_dropout,
                 relu_dropout,
                 emb_dropout,
                 res_dropout,
                 attn_mask,
                 scale_embedding=True):
        super(CrossmodalTransformer, self).__init__()
        self.attn_mask = attn_mask
        self.emb_scale = math.sqrt(d_model) if scale_embedding else 1.0
        self.pos_emb = SinusoidalPositionalEmbedding(d_model, 0, init_size=128)
        self.dropout = nn.Dropout(emb_dropout)

        layer = TransformerEncoderBlock(
            d_model=d_model,
            n_heads=n_heads,
            d_feedforward=d_model * 4,
            attn_dropout=attn_dropout,
            res_dropout=res_dropout,
            relu_dropout=relu_dropout
        )
        self.layers = _get_clones(layer, n_layers)

    def forward(self, x_query, x_key=None, key_mask=None):

        # query settings
        x_query_pos = self.pos_emb(x_query[:, :, 0])
        x_query = self.emb_scale * x_query + x_query_pos
        x_query = self.dropout(x_query).transpose(0, 1)

        # key settings
        if x_key is not None:
            x_key_pos = self.pos_emb(x_key[:, :, 0])
            x_key = self.emb_scale * x_key + x_key_pos
            x_key = self.dropout(x_key).transpose(0, 1)

        for layer in self.layers:
            x_query = layer(
                x_query, x_key,
                key_mask=key_mask,
                attn_mask=self.attn_mask
            )
        return x_query