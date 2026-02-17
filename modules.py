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

from timm.models.layers import to_2tuple
from torch.nn import init


class ReLU(nn.Hardtanh):

    def __init__(self, inplace=False):
        super(ReLU, self).__init__(0.0, 20.0, inplace)

    def __repr__(self):
        inplace_str = 'inplace' if self.inplace else ''
        return self.__class__.__name__ + ' (' \
            + inplace_str + ')'


def conv1x1(in_planes, out_planes, stride=1):
    "1x1 convolution without padding"
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     padding=0,
                     bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)

###########################################
# Commons
###########################################

class AttentiveStatisticsPooling(nn.Module):
    """
    Attentive Statistics Pooling (ASP)

    Input:
        x: Tensor of shape (B, C, T)
    Output:
        pooled: Tensor of shape (B, 2C)
    """
    def __init__(self, channels, hidden_channels=128, eps=1e-5):
        super().__init__()
        self.eps = eps

        self.attention = nn.Sequential(
            nn.Conv1d(channels, hidden_channels, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_channels),
            nn.Conv1d(hidden_channels, channels, kernel_size=1)
        )

    def forward(self, x):
        # x: (B, C, T)

        # Compute attention scores
        attn_scores = self.attention(x)          # (B, C, T)
        attn_weights = F.softmax(attn_scores, dim=2)

        # Weighted mean
        mean = torch.sum(attn_weights * x, dim=2)

        # Weighted standard deviation
        var = torch.sum(attn_weights * (x - mean.unsqueeze(2))**2, dim=2)
        std = torch.sqrt(var.clamp(min=self.eps))

        # Concatenate statistics
        pooled = torch.cat([mean, std], dim=1)   # (B, 2C)
        return pooled, attn_weights

class FrequencyAttention(nn.Module):
    def __init__(self, freq_dim, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(freq_dim, freq_dim // reduction),
            nn.ReLU(),
            nn.Linear(freq_dim // reduction, freq_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, C, T, F)
        attn = x.mean(dim=2)              # (B, C, F)
        attn = self.fc(attn)              # (B, C, F)
        attn = attn.unsqueeze(2)          # (B, C, 1, F)
        return x * attn, attn
    
class FeatureFrontend(nn.Module):
    def __init__(self, in_freq=80, out_dim=128):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.freq_attn = FrequencyAttention(freq_dim=in_freq)
        self.proj = nn.Linear(64 * in_freq, out_dim)

    def forward(self, x):
        # x: (B, T, F)
        B, T, F = x.shape

        x = x.unsqueeze(1)               # (B, 1, T, F)
        x = self.conv(x)                 # (B, 64, T, F)
        x, attn_weights = self.freq_attn(x)

        x = x.permute(0, 2, 1, 3)        # (B, T, 64, F)
        x = x.reshape(B, T, -1)          # (B, T, 64*F)

        x = self.proj(x)                 # (B, T, out_dim)
        return x, attn_weights


###########################################
# OPERAGT_MAE
###########################################

class PatchEmbed_org(nn.Module):
    """ Image to Patch Embedding"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        print(img_size,patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_hw = (img_size[1] // patch_size[1], img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        print('number of patches:', num_patches)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        #print('x shape:', x.shape)
        x = x.unsqueeze(1)
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        #assert H == self.img_size[0] and W == self.img_size[1], \
        #    f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        #y = self.proj(x)
        #print('y:', y.shape)
        x = self.proj(x).flatten(2).transpose(1, 2)
        #print('patch embedding:', x.shape)
        return x

class PatchEmbed_new(nn.Module):
    """ Flexible Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, stride=10):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        
        self.img_size = img_size
        self.patch_size = patch_size
        

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride) # with overlapped patches
        #self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        #self.patch_hw = (img_size[1] // patch_size[1], img_size[0] // patch_size[0])
        #self.num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        _, _, h, w = self.get_output_shape(img_size) # n, emb_dim, h, w
        self.patch_hw = (h, w)
        self.num_patches = h*w

    def get_output_shape(self, img_size):
        # todo: don't be lazy..
        return self.proj(torch.randn(1,1,img_size[0],img_size[1])).shape 

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        #assert H == self.img_size[0] and W == self.img_size[1], \
        #    f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        #x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.proj(x) # 32, 1, 1024, 128 -> 32, 768, 101, 12
        x = x.flatten(2) # 32, 768, 101, 12 -> 32, 768, 1212
        x = x.transpose(1, 2) # 32, 768, 1212 -> 32, 1212, 768
        return x

def get_2d_sincos_pos_embed_flexible(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

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
# ResnetManual module
# -----------------------------
def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
    
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class ChannelGateBAM(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16, num_layers=1):
        super(ChannelGateBAM, self).__init__()
        self.gate_c = nn.Sequential()
        self.gate_c.add_module( 'flatten', Flatten() )
        gate_channels = [gate_channel]
        gate_channels += [gate_channel // reduction_ratio] * num_layers
        gate_channels += [gate_channel]
        for i in range( len(gate_channels) - 2 ):
            self.gate_c.add_module( 'gate_c_fc_%d'%i, nn.Linear(gate_channels[i], gate_channels[i+1]) )
            self.gate_c.add_module( 'gate_c_bn_%d'%(i+1), nn.BatchNorm1d(gate_channels[i+1]) )
            self.gate_c.add_module( 'gate_c_relu_%d'%(i+1), nn.ReLU() )
        self.gate_c.add_module( 'gate_c_fc_final', nn.Linear(gate_channels[-2], gate_channels[-1]) )

    def forward(self, in_tensor):
        avg_pool = F.avg_pool2d( in_tensor, in_tensor.size(2), stride=in_tensor.size(2) )
        return self.gate_c( avg_pool ).unsqueeze(2).unsqueeze(3).expand_as(in_tensor)

class ChannelGateCBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGateCBAM, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

class SpatialGateBAM(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16, dilation_conv_num=2, dilation_val=4):
        super(SpatialGateBAM, self).__init__()
        self.gate_s = nn.Sequential()
        self.gate_s.add_module( 'gate_s_conv_reduce0', nn.Conv2d(gate_channel, gate_channel//reduction_ratio, kernel_size=1))
        self.gate_s.add_module( 'gate_s_bn_reduce0',	nn.BatchNorm2d(gate_channel//reduction_ratio) )
        self.gate_s.add_module( 'gate_s_relu_reduce0',nn.ReLU() )
        for i in range( dilation_conv_num ):
            self.gate_s.add_module( 'gate_s_conv_di_%d'%i, nn.Conv2d(gate_channel//reduction_ratio, gate_channel//reduction_ratio, kernel_size=3, \
						padding=dilation_val, dilation=dilation_val) )
            self.gate_s.add_module( 'gate_s_bn_di_%d'%i, nn.BatchNorm2d(gate_channel//reduction_ratio) )
            self.gate_s.add_module( 'gate_s_relu_di_%d'%i, nn.ReLU() )
        self.gate_s.add_module( 'gate_s_conv_final', nn.Conv2d(gate_channel//reduction_ratio, 1, kernel_size=1) )

    def forward(self, in_tensor):
        return self.gate_s( in_tensor ).expand_as(in_tensor)

class SpatialGateCBAM(nn.Module):
    def __init__(self):
        super(SpatialGateCBAM, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale
    
class BAM(nn.Module):
    def __init__(self, gate_channel):
        super(BAM, self).__init__()
        self.channel_att = ChannelGateBAM(gate_channel)
        self.spatial_att = SpatialGateBAM(gate_channel)
    def forward(self,in_tensor):
        att = 1 + F.sigmoid( self.channel_att(in_tensor) * self.spatial_att(in_tensor) )
        return att * in_tensor

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGateCBAM(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGateCBAM()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,
                          self.expansion * planes,
                          kernel_size=1,
                          stride=stride,
                          bias=False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes,
                               self.expansion * planes,
                               kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,
                          self.expansion * planes,
                          kernel_size=1,
                          stride=stride,
                          bias=False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class BasicBlockCBAM(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_cbam=False):
        super(BasicBlockCBAM, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        if use_cbam:
            self.cbam = CBAM( planes, 16 )
        else:
            self.cbam = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if not self.cbam is None:
            out = self.cbam(out)

        out += residual
        out = self.relu(out)

        return out

class BottleneckCBAM(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_cbam=False):
        super(BottleneckCBAM, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        if use_cbam:
            self.cbam = CBAM( planes * 4, 16 )
        else:
            self.cbam = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if not self.cbam is None:
            out = self.cbam(out)

        out += residual
        out = self.relu(out)

        return out


class TSTP(nn.Module):
    """
    Temporal statistics pooling, concatenate mean and std, which is used in
    x-vector
    Comment: simple concatenation can not make full use of both statistics
    """

    def __init__(self, in_dim=0, **kwargs):
        super(TSTP, self).__init__()
        self.in_dim = in_dim

    def forward(self, x):
        # The last dimension is the temporal axis
        pooling_mean = x.mean(dim=-1)
        pooling_std = torch.sqrt(torch.var(x, dim=-1) + 1e-7)
        pooling_mean = pooling_mean.flatten(start_dim=1)
        pooling_std = pooling_std.flatten(start_dim=1)
        stats = torch.cat((pooling_mean, pooling_std), 1)
        return stats

    def get_out_dim(self):
        self.out_dim = self.in_dim * 2
        return self.out_dim

class Resnet34Manual(nn.Module):
    def __init__(self, resnet_type, feature_dim, num_layers=4):
        super(Resnet34Manual, self).__init__()

        assert 1 <= num_layers <= 4, "num_layers must be between 1 and 4"

        if resnet_type == "resnet18":
            block = BasicBlock
            num_blocks = [2, 2, 2, 2]
        elif resnet_type == "resnet34":
            block = BasicBlock
            num_blocks = [3, 4, 6, 3]
        elif resnet_type == "resnet50":
            block = Bottleneck
            num_blocks = [3, 4, 6, 3]
        elif resnet_type == "resnet101":
            block = Bottleneck
            num_blocks = [3, 4, 23, 3]

        self.num_layers = num_layers
        m_channels = 32
        channel_mults = [1, 2, 4, 8]   # per-layer channel multipliers
        strides = [1, 2, 2, 2]          # per-layer strides

        def _downsample_freq(h: int, stages: int = 3) -> int:
            # Compute frequency downsampling after stride-2 stages precisely:
            # H_next = floor((H - 1) / 2) + 1 for k=3, p=1, s=2
            for _ in range(stages):
                h = (h - 1) // 2 + 1
            return h
        
        self.in_planes = m_channels
        self.feature_dim = feature_dim

        # Number of stride-2 stages depends on how many layers are used
        num_stride2_stages = max(0, num_layers - 1)  # layer1 has stride=1
        last_channels = m_channels * channel_mults[num_layers - 1]
        self.stats_dim = _downsample_freq(feature_dim, stages=num_stride2_stages) * last_channels

        self.conv1 = nn.Conv2d(1, m_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels)

        self.res_layers = nn.ModuleList()
        for i in range(num_layers):
            self.res_layers.append(
                self._make_layer(block, m_channels * channel_mults[i], num_blocks[i], stride=strides[i])
            )

        self.pool = TSTP(in_dim=self.stats_dim * block.expansion)

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
        out = F.relu(self.bn1(self.conv1(x)))
        for layer in self.res_layers:
            out = layer(out)
        stats = self.pool(out)
        return stats

class ResNetCBAM(nn.Module):
    def __init__(self, feature_dim, layers, network_type, att_type=None):
        super(ResNetCBAM, self).__init__()

        block = BasicBlockCBAM
        self.inplanes = 64
        self.network_type = network_type

        def _downsample_freq(h: int, stages: int = 3) -> int:
            # Compute frequency downsampling after 3 stride-2 stages precisely:
            # H_next = floor((H - 1) / 2) + 1 for k=3, p=1, s=2
            for _ in range(stages):
                h = (h - 1) // 2 + 1
            return h
        self.stats_dim = _downsample_freq(feature_dim) * 512 # 512 Last Layer

        if network_type == "ImageNet":
            self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            #self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.avgpool = nn.AvgPool2d(7)
        else:
            self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
            #self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        if att_type=='BAM':
            self.bam1 = BAM(64*block.expansion)
            self.bam2 = BAM(128*block.expansion)
            self.bam3 = BAM(256*block.expansion)
        else:
            self.bam1, self.bam2, self.bam3 = None, None, None

        self.layer1 = self._make_layer(block, 64,  layers[0], att_type=att_type)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, att_type=att_type)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, att_type=att_type)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, att_type=att_type)
        
        self.afterpooling = 512 * block.expansion

        for key in self.state_dict():
            if key.split('.')[-1]=="weight":
                if "conv" in key:
                    init.kaiming_normal(self.state_dict()[key], mode='fan_out')
                if "bn" in key:
                    if "SpatialGate" in key:
                        self.state_dict()[key][...] = 0
                    else:
                        self.state_dict()[key][...] = 1
            elif key.split(".")[-1]=='bias':
                self.state_dict()[key][...] = 0

    def _make_layer(self, block, planes, blocks, stride=1, att_type=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_cbam=att_type=='CBAM'))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_cbam=att_type=='CBAM'))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.network_type == "ImageNet":
            x = self.maxpool(x)

        x = self.layer1(x)
        if not self.bam1 is None:
            x = self.bam1(x)

        x = self.layer2(x)
        if not self.bam2 is None:
            x = self.bam2(x)

        x = self.layer3(x)
        if not self.bam3 is None:
            x = self.bam3(x)

        x = self.layer4(x)

        # if self.network_type == "ImageNet":
        #     x = self.avgpool(x)
        # else:
        #     x = F.avg_pool2d(x, 4)
        # x = x.view(x.size(0), -1)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        return x

class TemporalStatsPooling(nn.Module):
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, T, D)

        Returns:
            Tensor of shape (B, 2D) -> [mean | std]
        """
        mean = x.mean(dim=1)
        var = x.var(dim=1, unbiased=False)
        std = torch.sqrt(var + self.eps)

        return torch.cat([mean, std], dim=-1)
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