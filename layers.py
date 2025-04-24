import copy
import math
import numpy as np
import scipy
import torch
from torch import nn
from torch.nn import functional as F
from torch.cuda.amp import autocast, GradScaler

from modules import LayerNorm
import commons

class Pooling(nn.Module):
    def __init__(self):
        super().__init__()

    def compute_length_from_mask(self, mask):
        """
        Compute sequence lengths from the mask.
        mask: (batch_size, T)
        Assuming that the sampling rate is 16kHz, the frame shift is 20ms.
        """
        wav_lens = torch.sum(mask, dim=1)  # (batch_size, )
        feat_lens = torch.div(wav_lens - 1, 16000 * 0.02, rounding_mode="floor") + 1
        feat_lens = feat_lens.int().tolist()
        return feat_lens

    def forward(self, x, mask):
        raise NotImplementedError

class MultiheadAttentionPooling(Pooling):
    """
    MultiheadAttentionPooling: Combines multi-head attention and statistics pooling.
    """
    def __init__(self, input_size, num_heads):
        super().__init__()
        self._indim = input_size
        self.multihead_attn = nn.MultiheadAttention(embed_dim=input_size, num_heads=num_heads, batch_first=True)

    def forward(self, xs, mask):
        """
        xs: (batch_size, T, feat_dim)
        mask: (batch_size, T)

        => output: (batch_size, feat_dim * 2)
        """
        # Compute lengths from the mask
        feat_lens = self.compute_length_from_mask(mask)

        pooled_list = []
        for x, feat_len in zip(xs, feat_lens):
            x = x[:feat_len].unsqueeze(0)  # Keep only valid frames
            attended, _ = self.multihead_attn(x, x, x)

            # Compute statistics (mean and standard deviation)
            mu = attended.mean(dim=1)  # Mean pooling
            rh = torch.sqrt((attended.var(dim=1) + 1e-5))  # Standard deviation pooling

            # Concatenate statistics
            pooled = torch.cat((mu, rh), dim=-1)
            pooled_list.append(pooled.squeeze(0))

        return torch.stack(pooled_list)

class Encoder(nn.Module):
  def __init__(self, hidden_channels, filter_channels, n_heads, n_layers, kernel_size=1, p_dropout=0., window_size=None, block_length=None, gin_channels=0, emoin_channels=0, **kwargs):
    super().__init__()
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.window_size = window_size
    self.block_length = block_length
    self.gin_channels = gin_channels

    self.drop = nn.Dropout(p_dropout)
    self.attn_layers = nn.ModuleList()
    self.norm_layers_1 = nn.ModuleList()
    self.ffn_layers = nn.ModuleList()
    self.norm_layers_2 = nn.ModuleList()
    for i in range(self.n_layers):
      self.attn_layers.append(MultiHeadAttention(hidden_channels, hidden_channels, n_heads, window_size=window_size, p_dropout=p_dropout, block_length=block_length))
      self.norm_layers_1.append(LayerNorm(hidden_channels))
      self.ffn_layers.append(FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout=p_dropout))
      self.norm_layers_2.append(LayerNorm(hidden_channels))
    
    if gin_channels != 0:
      self.cond_g = nn.Linear(gin_channels, hidden_channels)

  def forward(self, x, x_mask, g=None, emo=None):
    attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
    x = x * x_mask
    
    for i in range(self.n_layers):
      if i == 3 - 1 and g is not None:
        x = x + self.cond_g(g.transpose(2, 1)).transpose(2, 1)

      y = self.attn_layers[i](x, x, attn_mask)
      y = self.drop(y)
      x = self.norm_layers_1[i](x + y)

      y = self.ffn_layers[i](x, x_mask)
      y = self.drop(y)
      x = self.norm_layers_2[i](x + y)
      
    x = x * x_mask
    return x

class MultiHeadAttention(nn.Module):
  def __init__(self, channels, out_channels, n_heads, window_size=None, heads_share=True, p_dropout=0., block_length=None, proximal_bias=False, proximal_init=False):
    super().__init__()
    assert channels % n_heads == 0

    self.channels = channels
    self.out_channels = out_channels
    self.n_heads = n_heads
    self.window_size = window_size
    self.heads_share = heads_share
    self.block_length = block_length
    self.proximal_bias = proximal_bias
    self.p_dropout = p_dropout
    self.attn = None

    self.k_channels = channels // n_heads
    self.conv_q = nn.Conv1d(channels, channels, 1)
    self.conv_k = nn.Conv1d(channels, channels, 1)
    self.conv_v = nn.Conv1d(channels, channels, 1)
    if window_size is not None:
      n_heads_rel = 1 if heads_share else n_heads
      rel_stddev = self.k_channels**-0.5
      self.emb_rel_k = nn.Parameter(torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev)
      self.emb_rel_v = nn.Parameter(torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev)
    self.conv_o = nn.Conv1d(channels, out_channels, 1)
    self.drop = nn.Dropout(p_dropout)

    nn.init.xavier_uniform_(self.conv_q.weight)
    nn.init.xavier_uniform_(self.conv_k.weight)
    if proximal_init:
      self.conv_k.weight.data.copy_(self.conv_q.weight.data)
      self.conv_k.bias.data.copy_(self.conv_q.bias.data)
    nn.init.xavier_uniform_(self.conv_v.weight)
      
  def forward(self, x, c, attn_mask=None):
    q = self.conv_q(x)
    k = self.conv_k(c)
    v = self.conv_v(c)
    
    x, self.attn = self.attention(q, k, v, mask=attn_mask)

    x = self.conv_o(x)
    return x
    
  def attention(self, query, key, value, mask=None):
    # reshape [b, d, t] -> [b, n_h, t, d_k]
    b, d, t_s, t_t = (*key.size(), query.size(2))
    query = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
    key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
    value = value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)

    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.k_channels)
    if self.window_size is not None:
      assert t_s == t_t, "Relative attention is only available for self-attention."
      key_relative_embeddings = self._get_relative_embeddings(self.emb_rel_k, t_s)
      rel_logits = self._matmul_with_relative_keys(query, key_relative_embeddings)
      rel_logits = self._relative_position_to_absolute_position(rel_logits)
      scores_local = rel_logits / math.sqrt(self.k_channels)
      scores = scores + scores_local
    if self.proximal_bias:
      assert t_s == t_t, "Proximal bias is only available for self-attention."
      scores = scores + self._attention_bias_proximal(t_s).to(device=scores.device, dtype=scores.dtype)
    if mask is not None:
      scores = scores.masked_fill(mask == 0, -1e4)
      if self.block_length is not None:
        block_mask = torch.ones_like(scores).triu(-self.block_length).tril(self.block_length)
        scores = scores * block_mask + -1e4*(1 - block_mask)
    p_attn = F.softmax(scores, dim=-1) # [b, n_h, t_t, t_s]
    p_attn = self.drop(p_attn)
    output = torch.matmul(p_attn, value)
    if self.window_size is not None:
      relative_weights = self._absolute_position_to_relative_position(p_attn)
      value_relative_embeddings = self._get_relative_embeddings(self.emb_rel_v, t_s)
      output = output + self._matmul_with_relative_values(relative_weights, value_relative_embeddings)
    output = output.transpose(2, 3).contiguous().view(b, d, t_t) # [b, n_h, t_t, d_k] -> [b, d, t_t]
    return output, p_attn

  def _matmul_with_relative_values(self, x, y):
    """
    x: [b, h, l, m]
    y: [h or 1, m, d]
    ret: [b, h, l, d]
    """
    ret = torch.matmul(x, y.unsqueeze(0))
    return ret

  def _matmul_with_relative_keys(self, x, y):
    """
    x: [b, h, l, d]
    y: [h or 1, m, d]
    ret: [b, h, l, m]
    """
    ret = torch.matmul(x, y.unsqueeze(0).transpose(-2, -1))
    return ret

  def _get_relative_embeddings(self, relative_embeddings, length):
    max_relative_position = 2 * self.window_size + 1
    # Pad first before slice to avoid using cond ops.
    pad_length = max(length - (self.window_size + 1), 0)
    slice_start_position = max((self.window_size + 1) - length, 0)
    slice_end_position = slice_start_position + 2 * length - 1
    if pad_length > 0:
      padded_relative_embeddings = F.pad(
          relative_embeddings,
          commons.convert_pad_shape([[0, 0], [pad_length, pad_length], [0, 0]]))
    else:
      padded_relative_embeddings = relative_embeddings
    used_relative_embeddings = padded_relative_embeddings[:,slice_start_position:slice_end_position]
    return used_relative_embeddings

  def _relative_position_to_absolute_position(self, x):
    """
    x: [b, h, l, 2*l-1]
    ret: [b, h, l, l]
    """
    batch, heads, length, _ = x.size()
    # Concat columns of pad to shift from relative to absolute indexing.
    x = F.pad(x, commons.convert_pad_shape([[0,0],[0,0],[0,0],[0,1]]))

    # Concat extra elements so to add up to shape (len+1, 2*len-1).
    x_flat = x.view([batch, heads, length * 2 * length])
    x_flat = F.pad(x_flat, commons.convert_pad_shape([[0,0],[0,0],[0,length-1]]))

    # Reshape and slice out the padded elements.
    x_final = x_flat.view([batch, heads, length+1, 2*length-1])[:, :, :length, length-1:]
    return x_final

  def _absolute_position_to_relative_position(self, x):
    """
    x: [b, h, l, l]
    ret: [b, h, l, 2*l-1]
    """
    batch, heads, length, _ = x.size()
    # padd along column
    x = F.pad(x, commons.convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, length-1]]))
    x_flat = x.view([batch, heads, length**2 + length*(length -1)])
    # add 0's in the beginning that will skew the elements after reshape
    x_flat = F.pad(x_flat, commons.convert_pad_shape([[0, 0], [0, 0], [length, 0]]))
    x_final = x_flat.view([batch, heads, length, 2*length])[:,:,:,1:]
    return x_final

  def _attention_bias_proximal(self, length):
    """Bias for self-attention to encourage attention to close positions.
    Args:
      length: an integer scalar.
    Returns:
      a Tensor with shape [1, 1, length, length]
    """
    r = torch.arange(length, dtype=torch.float32)
    diff = torch.unsqueeze(r, 0) - torch.unsqueeze(r, 1)
    return torch.unsqueeze(torch.unsqueeze(-torch.log1p(torch.abs(diff)), 0), 0)


class FFN(nn.Module):
  def __init__(self, in_channels, out_channels, filter_channels, kernel_size, p_dropout=0., activation=None):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.filter_channels = filter_channels
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.activation = activation

    self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size//2)
    self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size, padding=kernel_size//2)
    self.drop = nn.Dropout(p_dropout)

  def forward(self, x, x_mask):
    x = self.conv_1(x * x_mask)
    if self.activation == "gelu":
      x = x * torch.sigmoid(1.702 * x)
    else:
      x = torch.relu(x)
    x = self.drop(x)
    x = self.conv_2(x * x_mask)
    return x * x_mask

class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()
        self.demb = demb
        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.matmul(torch.unsqueeze(pos_seq, -1),
                                    torch.unsqueeze(self.inv_freq, 0))
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=1)
        if bsz is not None:
            return pos_emb[None, :, :].expand(bsz, -1, -1)
        else:
            return pos_emb[None, :, :]

class PositionwiseConvFF(nn.Module):
    def __init__(self, d_model, d_inner, kernel_size, dropout, sepconv=False, pre_lnorm=False):
        super(PositionwiseConvFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        if sepconv:
            self.conv_fn = SeparableConv
        else:
            self.conv_fn = nn.Conv1d

        self.CoreNet = nn.Sequential(
            self.conv_fn(d_model, d_inner, kernel_size, 1, (kernel_size // 2)),
            nn.ReLU(),
            # nn.Dropout(dropout),  # worse convergence
            self.conv_fn(d_inner, d_model, kernel_size, 1, (kernel_size // 2)),
            nn.Dropout(dropout),
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.pre_lnorm = pre_lnorm

    def forward(self, inp):
        return self._forward(inp)

    def _forward(self, inp):
        if self.pre_lnorm:
            # layer normalization + positionwise feed-forward
            core_out = inp.transpose(1, 2)
            core_out = self.CoreNet(self.layer_norm(core_out).to(inp.dtype))
            core_out = core_out.transpose(1, 2)

            # residual connection
            output = core_out + inp
        else:
            # positionwise feed-forward
            core_out = inp.transpose(1, 2)
            core_out = self.CoreNet(core_out)
            core_out = core_out.transpose(1, 2)

            # residual connection + layer normalization
            output = self.layer_norm(inp + core_out).to(inp.dtype)

        return output

class MultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0.1, pre_lnorm=False):
        super(MultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.scale = 1 / (d_head ** 0.5)
        self.pre_lnorm = pre_lnorm

        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head)
        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inp, attn_mask=None):
        return self._forward(inp, attn_mask)

    def _forward(self, inp, attn_mask=None):
        residual = inp

        if self.pre_lnorm:
            # layer normalization
            inp = self.layer_norm(inp)

        n_head, d_head = self.n_head, self.d_head

        head_q, head_k, head_v = torch.chunk(self.qkv_net(inp), 3, dim=2)
        head_q = head_q.view(inp.size(0), inp.size(1), n_head, d_head)
        head_k = head_k.view(inp.size(0), inp.size(1), n_head, d_head)
        head_v = head_v.view(inp.size(0), inp.size(1), n_head, d_head)

        q = head_q.permute(2, 0, 1, 3).reshape(-1, inp.size(1), d_head)
        k = head_k.permute(2, 0, 1, 3).reshape(-1, inp.size(1), d_head)
        v = head_v.permute(2, 0, 1, 3).reshape(-1, inp.size(1), d_head)

        attn_score = torch.bmm(q, k.transpose(1, 2))
        attn_score.mul_(self.scale)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).to(attn_score.dtype)
            attn_mask = attn_mask.repeat(n_head, attn_mask.size(2), 1)
            attn_score.masked_fill_(attn_mask.to(torch.bool), -float('inf'))

        attn_prob = F.softmax(attn_score, dim=2)
        attn_prob = self.dropatt(attn_prob)
        attn_vec = torch.bmm(attn_prob, v)

        attn_vec = attn_vec.view(n_head, inp.size(0), inp.size(1), d_head)
        attn_vec = attn_vec.permute(1, 2, 0, 3).contiguous().view(
            inp.size(0), inp.size(1), n_head * d_head)

        # linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = residual + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(residual + attn_out)

        output = output.to(attn_out.dtype)

        return output