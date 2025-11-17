import copy
import math
import numpy as np
import scipy
import torch
from torch import nn
from torch.nn import functional as F
from torch.cuda.amp import autocast, GradScaler
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from typing import Optional

def init_layer(layer):
    """Initialize a Linear or Convolutional layer."""
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.0)

def init_bn(bn):
    """Initialize a Batchnorm layer."""
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)

def init_gru(rnn):
    """Initialize a GRU layer."""

    def _concat_init(tensor, init_funcs):
        (length, fan_out) = tensor.shape
        fan_in = length // len(init_funcs)

        for i, init_func in enumerate(init_funcs):
            init_func(tensor[i * fan_in : (i + 1) * fan_in, :])

    def _inner_uniform(tensor):
        fan_in = nn.init._calculate_correct_fan(tensor, "fan_in")
        nn.init.uniform_(tensor, -math.sqrt(3 / fan_in), math.sqrt(3 / fan_in))

    for i in range(rnn.num_layers):
        _concat_init(
            getattr(rnn, "weight_ih_l{}".format(i)),
            [_inner_uniform, _inner_uniform, _inner_uniform],
        )
        torch.nn.init.constant_(getattr(rnn, "bias_ih_l{}".format(i)), 0)

        _concat_init(
            getattr(rnn, "weight_hh_l{}".format(i)),
            [_inner_uniform, _inner_uniform, nn.init.orthogonal_],
        )
        torch.nn.init.constant_(getattr(rnn, "bias_hh_l{}".format(i)), 0)

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        param in_channels: the number of input channels
        param out_channels: the number of out channels
        """
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type="avg"):
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == "max":
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg":
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg+max":
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception("Incorrect argument!")

        return x

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        """
        param channel: the number of input channels
        """
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        self.attn_vec = y
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# -----------------------------
# Cross-attention module
# -----------------------------
class BidirectionalCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.a2b = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.b2a = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.layernorm_a = nn.LayerNorm(dim)
        self.layernorm_b = nn.LayerNorm(dim)
        self.ffn_a = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim))
        self.ffn_b = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim))

    def forward(self, a, b, a_mask=None, b_mask=None):
        # a attends to b
        a_attn, _ = self.a2b(query=a, key=b, value=b, key_padding_mask=~b_mask if b_mask is not None else None)
        a = self.layernorm_a(a + a_attn)
        a = self.layernorm_a(a + self.ffn_a(a))

        # b attends to a
        b_attn, _ = self.b2a(query=b, key=a, value=a, key_padding_mask=~a_mask if a_mask is not None else None)
        b = self.layernorm_b(b + b_attn)
        b = self.layernorm_b(b + self.ffn_b(b))

        return a, b

# -----------------------------
# CRF sequence head wrapper
# -----------------------------
from torchcrf import CRF
class CRFSequenceHead(nn.Module):
    def __init__(self, hidden_dim, num_labels):
        super().__init__()
        self.emitter = nn.Linear(hidden_dim, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward_emissions(self, x):
        # x: [B, T, H]
        return self.emitter(x)  # emissions (unnormalized logits)

    def loss(self, emissions, tags, mask):
        # returns negative log-likelihood (mean)
        nll = -self.crf(emissions, tags, mask=mask, reduction='mean')
        return nll

    def decode(self, emissions, mask):
        return self.crf.decode(emissions, mask=mask)

# class LayerNorm(nn.Module):
#   def __init__(self, channels, eps=1e-4):
#       super().__init__()
#       self.channels = channels
#       self.eps = eps

#       self.gamma = nn.Parameter(torch.ones(channels))
#       self.beta = nn.Parameter(torch.zeros(channels))

#   def forward(self, x):
#     n_dims = len(x.shape)
#     mean = torch.mean(x, 1, keepdim=True)
#     variance = torch.mean((x -mean)**2, 1, keepdim=True)

#     x = (x - mean) * torch.rsqrt(variance + self.eps)

#     shape = [1, -1] + [1] * (n_dims - 2)
#     x = x * self.gamma.view(*shape) + self.beta.view(*shape)
#     return x

# class LayerNorm2(nn.Module):
#     def __init__(self, channels, eps=1e-4):
#         """Layer norm for the 2nd dimension of the input.
#         Args:
#             channels (int): number of channels (2nd dimension) of the input.
#             eps (float): to prevent 0 division

#         Shapes:
#             - input: (B, C, T)
#             - output: (B, C, T)
#         """
#         super().__init__()
#         self.channels = channels
#         self.eps = eps

#         self.gamma = nn.Parameter(torch.ones(1, channels, 1) * 0.1)
#         self.beta = nn.Parameter(torch.zeros(1, channels, 1))

#     def forward(self, x):
#         mean = torch.mean(x, 1, keepdim=True)
#         variance = torch.mean((x - mean) ** 2, 1, keepdim=True)
#         x = (x - mean) * torch.rsqrt(variance + self.eps)
#         x = x * self.gamma + self.beta
#         return x

# class ConvReluNorm(nn.Module):
#   def __init__(self, in_channels, hidden_channels, out_channels, kernel_size, n_layers, p_dropout):
#     super().__init__()
#     self.in_channels = in_channels
#     self.hidden_channels = hidden_channels
#     self.out_channels = out_channels
#     self.kernel_size = kernel_size
#     self.n_layers = n_layers
#     self.p_dropout = p_dropout
#     assert n_layers > 1, "Number of layers should be larger than 0."

#     self.conv_layers = nn.ModuleList()
#     self.norm_layers = nn.ModuleList()
#     self.conv_layers.append(nn.Conv1d(in_channels, hidden_channels, kernel_size, padding=kernel_size//2))
#     self.norm_layers.append(LayerNorm(hidden_channels))
#     self.relu_drop = nn.Sequential(
#         nn.ReLU(),
#         nn.Dropout(p_dropout))
#     for _ in range(n_layers-1):
#       self.conv_layers.append(nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size//2))
#       self.norm_layers.append(LayerNorm(hidden_channels))
#     self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
#     self.proj.weight.data.zero_()
#     self.proj.bias.data.zero_()

#   def forward(self, x, x_mask):
#     x_org = x
#     for i in range(self.n_layers):
#       x = self.conv_layers[i](x * x_mask)
#       x = self.norm_layers[i](x)
#       x = self.relu_drop(x)
#     x = x_org + self.proj(x)
#     return x * x_mask


# class ConvFastSpeech(nn.Module):
#     """
#     Convolution Module
#     """

#     def __init__(
#         self,
#         in_channels,
#         out_channels,
#         kernel_size=1,
#         stride=1,
#         padding=0,
#         dilation=1,
#         bias=True,
#         w_init="linear",
#     ):
#         """
#         :param in_channels: dimension of input
#         :param out_channels: dimension of output
#         :param kernel_size: size of kernel
#         :param stride: size of stride
#         :param padding: size of padding
#         :param dilation: dilation rate
#         :param bias: boolean. if True, bias is included.
#         :param w_init: str. weight inits with xavier initialization.
#         """
#         super(Conv, self).__init__()

#         self.conv = nn.Conv1d(
#             in_channels,
#             out_channels,
#             kernel_size=kernel_size,
#             stride=stride,
#             padding=padding,
#             dilation=dilation,
#             bias=bias,
#         )

#     def forward(self, x):
#         x = x.contiguous().transpose(1, 2)
#         x = self.conv(x)
#         x = x.contiguous().transpose(1, 2)

#         return x

# class DilatedDepthSeparableConv(nn.Module):
#     def __init__(self, channels, kernel_size, num_layers, dropout_p=0.0) -> torch.tensor:
#         """Dilated Depth-wise Separable Convolution module.

#         ::
#             x |-> DDSConv(x) -> LayerNorm(x) -> GeLU(x) -> Conv1x1(x) -> LayerNorm(x) -> GeLU(x) -> + -> o
#               |-------------------------------------------------------------------------------------^

#         Args:
#             channels ([type]): [description]
#             kernel_size ([type]): [description]
#             num_layers ([type]): [description]
#             dropout_p (float, optional): [description]. Defaults to 0.0.

#         Returns:
#             torch.tensor: Network output masked by the input sequence mask.
#         """
#         super().__init__()
#         self.num_layers = num_layers

#         self.convs_sep = nn.ModuleList()
#         self.convs_1x1 = nn.ModuleList()
#         self.norms_1 = nn.ModuleList()
#         self.norms_2 = nn.ModuleList()
#         for i in range(num_layers):
#             dilation = kernel_size**i
#             padding = (kernel_size * dilation - dilation) // 2
#             self.convs_sep.append(
#                 nn.Conv1d(channels, channels, kernel_size, groups=channels, dilation=dilation, padding=padding)
#             )
#             self.convs_1x1.append(nn.Conv1d(channels, channels, 1))
#             self.norms_1.append(LayerNorm2(channels))
#             self.norms_2.append(LayerNorm2(channels))
#         self.dropout = nn.Dropout(dropout_p)

#     def forward(self, x, x_mask, g=None):
#         """
#         Shapes:
#             - x: :math:`[B, C, T]`
#             - x_mask: :math:`[B, 1, T]`
#         """
#         if g is not None:
#             x = x + g
#         for i in range(self.num_layers):
#             y = self.convs_sep[i](x * x_mask)
#             y = self.norms_1[i](y)
#             y = F.gelu(y)
#             y = self.convs_1x1[i](y)
#             y = self.norms_2[i](y)
#             y = F.gelu(y)
#             y = self.dropout(y)
#             x = x + y
#         return x * x_mask

# class ElementwiseAffine(nn.Module):
#     """Element-wise affine transform like no-population stats BatchNorm alternative.

#     Args:
#         channels (int): Number of input tensor channels.
#     """

#     def __init__(self, channels):
#         super().__init__()
#         self.translation = nn.Parameter(torch.zeros(channels, 1))
#         self.log_scale = nn.Parameter(torch.zeros(channels, 1))

#     def forward(self, x, x_mask, reverse=False, **kwargs):  # pylint: disable=unused-argument
#         if not reverse:
#             y = (x * torch.exp(self.log_scale) + self.translation) * x_mask
#             logdet = torch.sum(self.log_scale * x_mask, [1, 2])
#             return y, logdet
#         x = (x - self.translation) * torch.exp(-self.log_scale) * x_mask
#         return x

# class ConvFlow(nn.Module):
#     """Dilated depth separable convolutional based spline flow.

#     Args:
#         in_channels (int): Number of input tensor channels.
#         hidden_channels (int): Number of in network channels.
#         kernel_size (int): Convolutional kernel size.
#         num_layers (int): Number of convolutional layers.
#         num_bins (int, optional): Number of spline bins. Defaults to 10.
#         tail_bound (float, optional): Tail bound for PRQT. Defaults to 5.0.
#     """

#     def __init__(
#         self,
#         in_channels: int,
#         hidden_channels: int,
#         kernel_size: int,
#         num_layers: int,
#         num_bins=10,
#         tail_bound=5.0,
#     ):
#         super().__init__()
#         self.num_bins = num_bins
#         self.tail_bound = tail_bound
#         self.hidden_channels = hidden_channels
#         self.half_channels = in_channels // 2

#         self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
#         self.convs = DilatedDepthSeparableConv(hidden_channels, kernel_size, num_layers, dropout_p=0.0)
#         self.proj = nn.Conv1d(hidden_channels, self.half_channels * (num_bins * 3 - 1), 1)
#         self.proj.weight.data.zero_()
#         self.proj.bias.data.zero_()

#     def forward(self, x, x_mask, g=None, reverse=False):
#         x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
#         h = self.pre(x0)
#         h = self.convs(h, x_mask, g=g)
#         h = self.proj(h) * x_mask

#         b, c, t = x0.shape
#         h = h.reshape(b, c, -1, t).permute(0, 1, 3, 2)  # [b, cx?, t] -> [b, c, t, ?]

#         unnormalized_widths = h[..., : self.num_bins] / math.sqrt(self.hidden_channels)
#         unnormalized_heights = h[..., self.num_bins : 2 * self.num_bins] / math.sqrt(self.hidden_channels)
#         unnormalized_derivatives = h[..., 2 * self.num_bins :]

#         x1, logabsdet = piecewise_rational_quadratic_transform(
#             x1,
#             unnormalized_widths,
#             unnormalized_heights,
#             unnormalized_derivatives,
#             inverse=reverse,
#             tails="linear",
#             tail_bound=self.tail_bound,
#         )

#         x = torch.cat([x0, x1], 1) * x_mask
#         logdet = torch.sum(logabsdet * x_mask, [1, 2])
#         if not reverse:
#             return x, logdet
#         return x
    
# class Mish(nn.Module):
#   def __init__(self):
#       super(Mish, self).__init__()
#   def forward(self, x):
#       return x * torch.tanh(F.softplus(x))


# # position wise encoding
# class PositionalEncodingComponent(nn.Module):
#     '''
#     Class to encode positional information to tokens.
#     For future, I want that this class to work even for sequences longer than 5000
#     '''

#     def __init__(self, hid_dim, dropout=0.2, max_len=5000):
#         super().__init__()

#         assert hid_dim % 2 == 0  # If not, it will result error in allocation to positional_encodings[:,1::2] later

#         self.dropout = nn.Dropout(dropout)

#         self.positional_encodings = nn.Parameter(torch.zeros(1, max_len, hid_dim), requires_grad=False)
#         # Positional Embeddings : [1,max_len,hid_dim]

#         pos = torch.arange(0, max_len).unsqueeze(1)  # pos : [max_len,1]
#         div_term = torch.exp(-torch.arange(0, hid_dim, 2) * math.log(
#             10000.0) / hid_dim)  # Calculating value of 1/(10000^(2i/hid_dim)) in log space and then exponentiating it
#         # div_term: [hid_dim//2]

#         self.positional_encodings[:, :, 0::2] = torch.sin(pos * div_term)  # pos*div_term [max_len,hid_dim//2]
#         self.positional_encodings[:, :, 1::2] = torch.cos(pos * div_term)

#     def forward(self, x):
#         # TODO: update this for very long sequences
#         x = x + self.positional_encodings[:, :x.size(1)].detach()
#         return self.dropout(x)


# # feed forward
# class FeedForwardComponent(nn.Module):
#     '''
#     Class for pointwise feed forward connections
#     '''

#     def __init__(self, hid_dim, pf_dim, dropout):
#         super().__init__()

#         self.dropout = nn.Dropout(dropout)

#         self.fc1 = nn.Linear(hid_dim, pf_dim)
#         self.fc2 = nn.Linear(pf_dim, hid_dim)

#     def forward(self, x):
#         # x : [batch_size,seq_len,hid_dim]
#         x = self.dropout(torch.relu(self.fc1(x)))

#         # x : [batch_size,seq_len,pf_dim]
#         x = self.fc2(x)

#         # x : [batch_size,seq_len,hid_dim]
#         return x


# # multi headed attention
# class MultiHeadedAttentionComponent(nn.Module):
#     '''
#     Multiheaded attention Component.
#     '''

#     def __init__(self, hid_dim, n_heads, dropout):
#         super().__init__()

#         assert hid_dim % n_heads == 0  # Since we split hid_dims into n_heads

#         self.hid_dim = hid_dim
#         self.n_heads = n_heads  # no of heads in 'multiheaded' attention
#         self.head_dim = hid_dim // n_heads  # dims of each head

#         # Transformation from source vector to query vector
#         self.fc_q = nn.Linear(hid_dim, hid_dim)

#         # Transformation from source vector to key vector
#         self.fc_k = nn.Linear(hid_dim, hid_dim)

#         # Transformation from source vector to value vector
#         self.fc_v = nn.Linear(hid_dim, hid_dim)

#         self.fc_o = nn.Linear(hid_dim, hid_dim)

#         self.dropout = nn.Dropout(dropout)

#         # Used in self attention for smoother gradients
#         self.scale = nn.Parameter(torch.sqrt(torch.FloatTensor([self.head_dim])), requires_grad=False)

#     def forward(self, query, key, value, mask: Optional[torch.Tensor] = None):
#         # query : [batch_size, query_len, hid_dim]
#         # key : [batch_size, key_len, hid_dim]
#         # value : [batch_size, value_len, hid_dim]

#         batch_size = query.shape[0]

#         # Transforming quey,key,values
#         Q = self.fc_q(query)
#         K = self.fc_k(key)
#         V = self.fc_v(value)

#         # Q : [batch_size, query_len, hid_dim]
#         # K : [batch_size, key_len, hid_dim]
#         # V : [batch_size, value_len,hid_dim]

#         # Changing shapes to acocmadate n_heads information
#         Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
#         K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
#         V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

#         # Q : [batch_size, n_heads, query_len, head_dim]
#         # K : [batch_size, n_heads, key_len, head_dim]
#         # V : [batch_size, n_heads, value_len, head_dim]

#         # Calculating alpha
#         score = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
#         # score : [batch_size, n_heads, query_len, key_len]

#         if mask is not None:
#             score = score.masked_fill(mask == 0, -1e10)

#         alpha = torch.softmax(score, dim=-1)
#         # alpha : [batch_size, n_heads, query_len, key_len]

#         # Get the final self-attention  vector
#         x = torch.matmul(self.dropout(alpha), V)
#         # x : [batch_size, n_heads, query_len, head_dim]

#         # Reshaping self attention vector to concatenate
#         x = x.permute(0, 2, 1, 3).contiguous()
#         # x : [batch_size, query_len, n_heads, head_dim]

#         x = x.view(batch_size, -1, self.hid_dim)
#         # x: [batch_size, query_len, hid_dim]

#         # Transforming concatenated outputs
#         x = self.fc_o(x)
#         # x : [batch_size, query_len, hid_dim]

#         return x, alpha


# # EncodingLayer
# class EncodingLayer(nn.Module):
#     '''
#     Operations of a single layer. Each layer contains:
#     1) multihead attention, followed by
#     2) LayerNorm of addition of multihead attention output and input to the layer, followed by
#     3) FeedForward connections, followed by
#     4) LayerNorm of addition of FeedForward outputs and output of previous layerNorm.
#     '''

#     def __init__(self, hid_dim, n_heads, pf_dim, dropout):
#         super().__init__()

#         self.self_attn_layer_norm = nn.LayerNorm(hid_dim)  # Layer norm after self-attention
#         self.ff_layer_norm = nn.LayerNorm(hid_dim)  # Layer norm after FeedForward component

#         self.self_attention = MultiHeadedAttentionComponent(hid_dim, n_heads, dropout)
#         self.feed_forward = FeedForwardComponent(hid_dim, pf_dim, dropout)

#         self.dropout = nn.Dropout(dropout)

#     def forward(self, src, src_mask=None):
#         # src : [batch_size, src_len, hid_dim]
#         # src_mask : [batch_size, 1, 1, src_len]

#         # get self-attention
#         _src, _ = self.self_attention(src, src, src, src_mask)

#         # LayerNorm after dropout
#         src = self.self_attn_layer_norm(src + self.dropout(_src))
#         # src : [batch_size, src_len, hid_dim]

#         # FeedForward
#         _src = self.feed_forward(src)

#         # layerNorm after dropout
#         src = self.ff_layer_norm(src + self.dropout(_src))
#         # src: [batch_size, src_len, hid_dim]

#         return src
    
# def clean_state_dict(state_dict):
#     new = {}
#     for key, value in state_dict.items():
#         if key in ['fc.weight', 'fc.bias']:
#             continue
#         new[key.replace('bert.', '')] = value
#     return new

# class TransformerBlock(nn.Module, ABC):
#     def __init__(self,
#                  d_model,
#                  n_heads,
#                  attn_dropout,
#                  res_dropout):
#         super(TransformerBlock, self).__init__()
#         self.layer_norm = nn.LayerNorm(d_model)
#         self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=attn_dropout)
#         self.dropout = nn.Dropout(res_dropout)

#     def forward(self,
#                 query, key, value,
#                 key_padding_mask=None,
#                 attn_mask=True):
#         """
#         From original Multimodal Transformer code,

#         In the original paper each operation (multi-head attention or FFN) is
#         post-processed with: `dropout -> add residual -> layer-norm`. In the
#         tensor2tensor code they suggest that learning is more robust when
#         preprocessing each layer with layer-norm and postprocessing with:
#         `dropout -> add residual`. We default to the approach in the paper.
#         """
#         query, key, value = [self.layer_norm(x) for x in (query, key, value)]
#         mask = self.get_future_mask(query, key) if attn_mask else None
#         x = self.self_attn(
#             query, key, value,
#             key_padding_mask=key_padding_mask,
#             attn_mask=mask)[0]
#         return query + self.dropout(x)

#     @staticmethod
#     def get_future_mask(query, key=None):
#         """
#         :return: source mask
#             ex) tensor([[0., -inf, -inf],
#                         [0., 0., -inf],
#                         [0., 0., 0.]])
#         """
#         dim_query = query.shape[0]
#         dim_key = dim_query if key is None else key.shape[0]

#         future_mask = torch.ones(dim_query, dim_key, device=query.device)
#         future_mask = torch.triu(future_mask, diagonal=1).float()
#         future_mask = future_mask.masked_fill(future_mask == float(1), float('-inf'))
#         return future_mask


# class FeedForwardBlock(nn.Module, ABC):
#     def __init__(self,
#                  d_model,
#                  d_feedforward,
#                  res_dropout,
#                  relu_dropout):
#         super(FeedForwardBlock, self).__init__()
#         self.layer_norm = nn.LayerNorm(d_model)
#         self.linear1 = nn.Linear(d_model, d_feedforward)
#         self.dropout1 = nn.Dropout(relu_dropout)
#         self.linear2 = nn.Linear(d_feedforward, d_model)
#         self.dropout2 = nn.Dropout(res_dropout)

#     def forward(self, x):
#         """
#         Do layer-norm before self-attention
#         """
#         normed = self.layer_norm(x)
#         projected = self.linear2(self.dropout1(F.relu(self.linear1(normed))))
#         skipped = normed + self.dropout2(projected)
#         return skipped


# class TransformerEncoderBlock(nn.Module, ABC):
#     def __init__(self,
#                  d_model,
#                  n_heads,
#                  d_feedforward,
#                  attn_dropout,
#                  res_dropout,
#                  relu_dropout):
#         """
#         Args:
#             d_model: the number of expected features in the input (required).
#             n_heads: the number of heads in the multi-head attention models (required).
#             d_feedforward: the dimension of the feedforward network model (required).
#             attn_dropout: the dropout value for multi-head attention (required).
#             res_dropout: the dropout value for residual connection (required).
#             relu_dropout: the dropout value for relu (required).
#         """
#         super(TransformerEncoderBlock, self).__init__()
#         self.transformer = TransformerBlock(d_model, n_heads, attn_dropout, res_dropout)
#         self.feedforward = FeedForwardBlock(d_model, d_feedforward, res_dropout, relu_dropout)

#     def forward(self,
#                 x_query,
#                 x_key=None,
#                 key_mask=None,
#                 attn_mask=None):
#         """
#         x : input of the encoder layer -> (L, B, d)
#         """
#         if x_key is not None:
#             x = self.transformer(
#                 x_query, x_key, x_key,
#                 key_padding_mask=key_mask,
#                 attn_mask=attn_mask
#             )
#         else:
#             x = self.transformer(
#                 x_query, x_query, x_query,
#                 key_padding_mask=key_mask,
#                 attn_mask=attn_mask
#             )
#         x = self.feedforward(x)
#         return x

# class ConvBlock(nn.Module):
#     """
#     Convolutional block with Conv2D -> BatchNorm -> ReLU -> MaxPool -> Dropout.
#     Designed for feature extraction from audio input.
#     """
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, pool_size=(5, 4), dropout=0.05):
#         super().__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.pool = nn.MaxPool2d(pool_size)
#         self.dropout = nn.Dropout2d(dropout)

#     def forward(self, x):
#         x = F.relu(self.bn(self.conv(x)))
#         x = self.pool(x)
#         x = self.dropout(x)
#         return x
# #########################
# #
# #          VIT
# #
# #########################

# class FeedForwardVIT(nn.Module):
#     def __init__(self, dim, hidden_dim, dropout = 0.):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.LayerNorm(dim),
#             nn.Linear(dim, hidden_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, dim),
#             nn.Dropout(dropout)
#         )

#     def forward(self, x):
#         return self.net(x)
    
# class AttentionVIT(nn.Module):
#     def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
#         super().__init__()
#         inner_dim = dim_head *  heads
#         project_out = not (heads == 1 and dim_head == dim)

#         self.heads = heads
#         self.scale = dim_head ** -0.5

#         self.norm = nn.LayerNorm(dim)

#         self.attend = nn.Softmax(dim = -1)
#         self.dropout = nn.Dropout(dropout)

#         self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

#         self.to_out = nn.Sequential(
#             nn.Linear(inner_dim, dim),
#             nn.Dropout(dropout)
#         ) if project_out else nn.Identity()

#     def forward(self, x):
#         x = self.norm(x)

#         qkv = self.to_qkv(x).chunk(3, dim = -1)
#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

#         dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

#         attn = self.attend(dots)
#         attn = self.dropout(attn)

#         out = torch.matmul(attn, v)
#         out = rearrange(out, 'b h n d -> b n (h d)')
#         return self.to_out(out)
    
# class TransformerVIT(nn.Module):
#     def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
#         super().__init__()
#         self.norm = nn.LayerNorm(dim)
#         self.layers = nn.ModuleList([])
#         for _ in range(depth):
#             self.layers.append(nn.ModuleList([
#                 AttentionVIT(dim, heads = heads, dim_head = dim_head, dropout = dropout),
#                 FeedForwardVIT(dim, mlp_dim, dropout = dropout)
#             ]))

#     def forward(self, x):
#         for attn, ff in self.layers:
#             x = attn(x) + x
#             x = ff(x) + x

#         return self.norm(x)