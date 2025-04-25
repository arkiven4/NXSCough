import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import layers
import commons
import modules

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TDNNLayer(nn.Module):
    def __init__(self, tdnn_dim, tdnn_kernel, tdnn_dilation, layer_id=0):
        super().__init__()
        self.in_conv_dim = tdnn_dim[layer_id - 1] if layer_id > 0 else tdnn_dim[layer_id]
        self.out_conv_dim = tdnn_dim[layer_id]
        self.kernel_size = tdnn_kernel[layer_id]
        self.dilation = tdnn_dilation[layer_id]

        self.kernel = nn.Linear(self.in_conv_dim * self.kernel_size, self.out_conv_dim)
        self.activation = nn.ReLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.transpose(1, 2)
        weight = self.kernel.weight.view(self.out_conv_dim, self.kernel_size, self.in_conv_dim).transpose(1, 2)
        hidden_states = nn.functional.conv1d(hidden_states, weight, self.kernel.bias, dilation=self.dilation)
        hidden_states = hidden_states.transpose(1, 2)

        hidden_states = self.activation(hidden_states)
        return hidden_states

class DimensionPredictorVITS(nn.Module):
  def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, out_channel=1, gin_channels=0, txtin_channels=0):
    super().__init__()

    self.in_channels = in_channels
    self.filter_channels = filter_channels
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.gin_channels = gin_channels
    self.txtin_channels = txtin_channels

    self.drop = nn.Dropout(p_dropout)
    self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size//2)
    self.norm_1 = modules.LayerNorm(filter_channels)
    self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size//2)
    self.norm_2 = modules.LayerNorm(filter_channels)
    self.proj = nn.Conv1d(filter_channels, out_channel, 1)

    if gin_channels != 0:
      self.cond = nn.Conv1d(gin_channels, in_channels, 1)

    if txtin_channels != 0:
      self.cond_txt = nn.Conv1d(txtin_channels, in_channels, 1)

  def forward(self, x, x_mask, g=None, txt=None):
    if g is not None:
      g = torch.detach(g)
      x = x + self.cond(g)

    if txt is not None:
      txt = torch.detach(txt)
      x = x + self.cond_txt(txt)

    x = self.conv_1(x * x_mask)
    x = torch.relu(x)
    x = self.norm_1(x)
    x = self.drop(x)
    x = self.conv_2(x * x_mask)
    x = torch.relu(x)
    x = self.norm_2(x)
    x = self.drop(x)
    x = self.proj(x * x_mask)
    return x * x_mask

class TransformerLayer(nn.Module):
    def __init__(self, d_dim, d_dim2, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=d_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_dim)
        
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_dim, kdim=d_dim2, vdim=d_dim2, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, d_dim)
        )
        self.norm3 = nn.LayerNorm(d_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask, key, value):
        # Self-Attention
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Cross-Attention
        cross_attn_out, _ = self.cross_attn(x, key, value)
        x = self.norm2(x + self.dropout(cross_attn_out))
        
        # Feed Forward
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))   
        return x

class AttentionPooling(nn.Module):

    """
        Sequence to one component, the input dimension is the same than the output dimension.
        Sequence length is not fixed.
        Given n vectors, takes their weighted average as output. These weights comes from an attention mechanism.
        It can be seen as a One Head Self-Attention, where a unique query is used and input vectors are the values and keys.   
        emb_in is the dimension of every input vector (embedding).
    """

    def __init__(self, emb_in):

        super().__init__()

        self.emb_in = emb_in
        self.init_query()
        
    def init_query(self):
        self.query = torch.nn.Parameter(torch.FloatTensor(self.emb_in, 1))
        torch.nn.init.xavier_normal_(self.query)


    def forward(self, input_tensors):
        b, t, e = input_tensors.size()
        assert e == self.emb_in, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb_in})'
        attention_scores = torch.matmul(input_tensors, self.query)
        attention_scores = attention_scores.squeeze(dim = -1)
        attention_scores = F.softmax(attention_scores, dim = 1)
        attention_scores = attention_scores.unsqueeze(dim = -1)
        output = torch.bmm(attention_scores.transpose(1, 2), input_tensors)
        output = output.view(output.size()[0], output.size()[1] * output.size()[2])
        
        return output

class AttentiveStatisticsPooling(nn.Module):
    """
    AttentiveStatisticsPooling
    Paper: Attentive Statistics Pooling for Deep Speaker Embedding
    Link: https://arxiv.org/pdf/1803.10963.pdf
    """
    def __init__(self, input_size):
        super().__init__()
        self._indim = input_size
        self.sap_linear = nn.Linear(input_size, input_size)
        self.attention = nn.Parameter(torch.FloatTensor(input_size, 1))
        torch.nn.init.normal_(self.attention, mean=0, std=1)

    def forward(self, xs, feat_lens):
        """
        xs: (batch_size, T, feat_dim)
        mask: (batch_size, T)

        => output: (batch_size, feat_dim*2)
        """
        pooled_list = []
        for x, feat_len in zip(xs, feat_lens):
            x = x[:feat_len].unsqueeze(0)
            h = torch.tanh(self.sap_linear(x))
            w = torch.matmul(h, self.attention).squeeze(dim=2)
            w = F.softmax(w, dim=1).view(x.size(0), x.size(1), 1)
            mu = torch.sum(x * w, dim=1)
            rh = torch.sqrt((torch.sum((x**2) * w, dim=1) - mu**2).clamp(min=1e-5))
            x = torch.cat((mu, rh), 1).squeeze(0)
            pooled_list.append(x)
        return torch.stack(pooled_list)

#######################################
#######################################
#######################################
#######################################
#######################################
#######################################

class SimplePooling2(nn.Module):
    def __init__(self, ssl_hidden_channels, filter_channels, d_dim,
                       n_heads, n_layers, pooling_hidden,
                       kernel_size, p_dropout, temporal_pred="GlowDP", **kwargs):
        super(SimplePooling2, self).__init__()
        
        self.eps = 1e-5
        self.temporal_pred = temporal_pred
        self.ssl_hidden_channels = ssl_hidden_channels

        self.layer_norm_x = nn.LayerNorm(ssl_hidden_channels)
        self.x_pooling = AttentiveStatisticsPooling(ssl_hidden_channels)

    def forward(self, x, x_lengths, attention_mask=None): # x [b, 2028] or [11, 25, 406, 1024])
        hidden_states = x
        hidden_states = self.layer_norm_x(hidden_states)

        x_pooled = self.x_pooling(hidden_states, x_lengths)

        decoder_out = x_pooled #torch.cat([x_pooled, txt_pooled], dim=1)
        return decoder_out
