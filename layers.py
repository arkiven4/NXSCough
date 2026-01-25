import torch, math
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.wavlm.modeling_wavlm import WavLMGroupNormConvLayer, WavLMNoLayerNormConvLayer, WavLMLayerNormConvLayer
import modules

class WavLMFeatureEncoder(nn.Module):
    """Construct the features from raw audio waveform"""

    def __init__(self, config):
        super().__init__()
        
        if config.feat_extract_norm == "group":
            conv_layers = [WavLMGroupNormConvLayer(config, layer_id=0)] + [
                WavLMNoLayerNormConvLayer(config, layer_id=i + 1) for i in range(config.num_feat_extract_layers - 1)
            ]
        elif config.feat_extract_norm == "layer":
           conv_layers = [WavLMLayerNormConvLayer(config, layer_id=i) for i in range(config.num_feat_extract_layers)]
        else:
            raise ValueError(
                f"`config.feat_extract_norm` is {config.feat_extract_norm}, but has to be one of ['group', 'layer']"
            )
        self.conv_layers = nn.ModuleList(conv_layers)
        self.gradient_checkpointing = False
        self._requires_grad = True

    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def forward(self, input_values):
        hidden_states = input_values[:, None]

        # make sure hidden_states require grad for gradient_checkpointing
        # if self._requires_grad and self.training:
        #     hidden_states.requires_grad = True

        for conv_layer in self.conv_layers:
            hidden_states = conv_layer(hidden_states)

        return hidden_states
    

class WavLMFeatureProjection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.conv_dim[-1], eps=config.layer_norm_eps)
        self.projection = nn.Linear(config.conv_dim[-1], config.hidden_size)
        self.dropout = nn.Dropout(config.feat_proj_dropout)

    def forward(self, hidden_states):
        # non-projected hidden states are needed for quantization
        norm_hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.projection(norm_hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states, norm_hidden_states
    

class WhisperFeatureEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        self.conv1 = nn.Conv1d(self.num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)

        self.embed_positions = nn.Embedding(self.max_source_positions, embed_dim)
        self.embed_positions.requires_grad_(False)

        self.gradient_checkpointing = False

    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def forward(self, input_features):
        expected_seq_length = self.config.max_source_positions * self.conv1.stride[0] * self.conv2.stride[0]
        if input_features.shape[-1] != expected_seq_length:
            raise ValueError(
                f"Whisper expects the mel input features to be of length {expected_seq_length}, but found {input_features.shape[-1]}. Make sure to pad the input mel features to {expected_seq_length}."
            )

        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))

        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        all_positions = torch.arange(self.embed_positions.num_embeddings, device=inputs_embeds.device)

        hidden_states = inputs_embeds + self.embed_positions(all_positions)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        return hidden_states

class AttentiveStatisticsPooling(nn.Module):
    """Attentive Statistics Pooling layer that computes weighted mean and std
    attention_dim	Behavior
    64	lightweight, low-capacity attention
    128	baseline, balanced (recommended)
    256	high-capacity attention, improved discrimination
    512	very expressive but risk of overfitting; slower
    """
    def __init__(self, input_dim, attention_dim=128):
        super(AttentiveStatisticsPooling, self).__init__()
        self.input_dim = input_dim
        self.attention_dim = attention_dim
        
        # Attention mechanism
        self.attention_layer = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1, bias=False)
        )
        
    def forward(self, x, lengths=None):
        """
        Args:
            x: Input tensor of shape [B, T, D] where B=batch, D=features, T=time
            lengths: Optional tensor of actual sequence lengths for masking
        Returns:
            pooled: Concatenated mean and std of shape [B, 2*D]
        """
        batch_size, time_steps, feature_dim = x.shape
        
        # Compute attention weights
        attention_weights = self.attention_layer(x)  # [B, T, 1]
        
        # Apply length masking if provided
        if lengths is not None:
            mask = torch.arange(time_steps, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
            mask = mask.unsqueeze(-1).float()  # [B, T, 1]
            attention_weights = attention_weights.masked_fill(mask == 0, float('-inf'))
        
        # Softmax to get normalized attention weights
        attention_weights = F.softmax(attention_weights, dim=1)  # [B, T, 1]
        
        # Compute weighted statistics
        # Weighted mean
        weighted_mean = torch.sum(attention_weights * x, dim=1)  # [B, D]
        
        # Weighted standard deviation
        diff = x - weighted_mean.unsqueeze(1)  # [B, T, D]
        weighted_var = torch.sum(attention_weights * diff**2, dim=1)  # [B, D]
        weighted_std = torch.sqrt(weighted_var + 1e-8)  # [B, D]
        
        # Concatenate mean and std
        pooled = torch.cat([weighted_mean, weighted_std], dim=1)  # [B, 2*D]
        
        return pooled  

class SEBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        param in_channels: the number of input channels
        param out_channels: the number of out channels
        """
        super(SEBlock, self).__init__()

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

        self.se1 = modules.SELayer(out_channels)
        self.se2 = modules.SELayer(out_channels)

        self.init_weight()

    def init_weight(self):
        modules.init_layer(self.conv1)
        modules.init_layer(self.conv2)
        modules.init_bn(self.bn1)
        modules.init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type="avg"):
        x = input
        x = F.relu_(self.se1(self.bn1(self.conv1(x))))
        x = F.relu_(self.se2(self.bn2(self.conv2(x))))
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

class CustomWalvmProjector(nn.Module):
    def __init__(self, dim_in=1024, dim_hidden=2048, dim_out=256):
        super().__init__()
        # CHANGE: explicit layer definitions to ensure final linear+BN are executed
        self.fc1 = nn.Linear(dim_in, dim_hidden)
        self.bn1 = nn.BatchNorm1d(dim_hidden)
        self.fc2 = nn.Linear(dim_hidden, dim_hidden)
        self.bn2 = nn.BatchNorm1d(dim_hidden)

        # residual mapping (dim_in -> dim_hidden)
        self.shortcut = nn.Linear(dim_in, dim_hidden) if dim_in != dim_hidden else nn.Identity()

        # final projection to dim_out + BN (affine=False)
        self.fc3 = nn.Linear(dim_hidden, dim_out)
        self.bn3 = nn.BatchNorm1d(dim_out, affine=False)

    def forward(self, x):
        # x: [B, dim_in]
        # block 1
        out = self.fc1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.fc2(out)
        out = self.bn2(out)
        out = F.relu(out, inplace=True)

        # residual add (CHANGE: residual applied before dimension collapse)
        res = self.shortcut(x) if not isinstance(self.shortcut, nn.Identity) else x
        out = out + res

        # final projection
        out = self.fc3(out)
        out = self.bn3(out)  # affine False -> whitened output (SimCLR v2 style)

        # optional: L2 normalize here or leave to loss function (we normalize in calc_cl)
        return out
    

class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""
    # https://github.dev/lyakaap/NetVLAD-pytorch/blob/master/netvlad.py
    def __init__(self, num_clusters=64, dim=128, alpha=100.0,
                 normalize_input=True):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = alpha
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=True)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        self._init_params()

    def _init_params(self):
        self.conv.weight = nn.Parameter(
            (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
        )
        self.conv.bias = nn.Parameter(
            - self.alpha * self.centroids.norm(dim=1)
        )

    def forward(self, x):
        B, C = x.shape[:2]

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        # soft-assignment
        soft_assign = self.conv(x).view(B, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)

        x_flatten = x.view(B, C, -1)
        
        # calculate residuals to each clusters
        residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
            self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
        residual *= soft_assign.unsqueeze(2)
        vlad = residual.sum(dim=-1)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        return vlad

class BasicBlockRes2Net(nn.Module):
    expansion = 2
    def __init__(self, in_planes, planes, stride=1, baseWidth=32, scale=2):
        super(BasicBlockRes2Net, self).__init__()
        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = modules.conv1x1(in_planes, width * scale, stride)
        self.bn1 = nn.BatchNorm2d(width * scale)
        self.nums = scale - 1
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(modules.conv3x3(width, width))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.relu = modules.ReLU(inplace=True)

        self.conv3 = modules.conv1x1(width * scale, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,
                          self.expansion * planes,
                          kernel_size=1,
                          stride=stride,
                          bias=False), nn.BatchNorm2d(self.expansion * planes))
        self.stride = stride
        self.width = width
        self.scale = scale

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        spx = torch.split(out, self.width, 1)
        sp = spx[0]
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            if i >= 1:
                sp = sp + spx[i]
            sp = conv(sp)
            sp = self.relu(bn(sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)

        out = torch.cat((out, spx[self.nums]), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        residual = self.shortcut(x)
        out += residual
        out = self.relu(out)

        return out