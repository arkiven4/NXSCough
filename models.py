import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import torchvision

from typing import Any, Callable, Optional

import modules, commons

class LSTMAudioClassifier1(nn.Module):
    def __init__(self, input_size, seq_len, hidden_size, num_classes):
        super(LSTMAudioClassifier1, self).__init__()
        
        self.batch_norm1 = nn.BatchNorm1d(input_size)
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        
        #self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8, batch_first=True)

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, lengths=None):
        x = self.batch_norm1(x.transpose(1, 2)).transpose(1, 2)
        x, _ = self.lstm1(x)
        
        x = self.batch_norm2(x.transpose(1, 2)).transpose(1, 2)
        x, _ = self.lstm2(x)
        
        # attn_output, _ = self.attention(x, x, x)
        # x = torch.mean(attn_output, dim=1)

        x = self.flatten(x[:, -1, :])
        x = self.dropout(x)
        x = self.fc(x)
        return x

class ResNet101(torchvision.models.resnet.ResNet):
    def __init__(self, dummy, output_dim, track_bn=True, **kwargs):
        def norm_layer(*args, **kwargs):
            return nn.BatchNorm2d(*args, **kwargs, track_running_stats=track_bn)
        super().__init__(torchvision.models.resnet.Bottleneck, [3, 4, 23, 3], norm_layer=norm_layer, num_classes=output_dim)
        #del self.fc
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.final_feat_dim = 2048
        self.grad_cam = False
         # TODO : Coba tambah Rezize and Normalization
        self.preprocess = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224,224))
        ])

    def load_sl_official_weights(self, progress=True):
        state_dict = load_state_dict_from_url(torchvision.models.resnet.ResNet101_Weights.IMAGENET1K_V2.url,
                                              progress=progress)

        del state_dict['conv1.weight']
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        # if len(missing) > 0:
            # raise AssertionError('Model code may be incorrect')

    def load_ssl_official_weights(self, progress=True):
        raise NotImplemented

    def _forward_impl(self, x: Tensor, lengths=None) -> Tensor:
        # See note [TorchScript super()]
        if self.grad_cam == True:
            x = x
        else:
            x = x.unsqueeze(0)
            x = self.preprocess(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
    
class InceptionV3(torchvision.models.inception.Inception3):
    def __init__(self, dummy, output_dim, track_bn=True, **kwargs):
        super().__init__(num_classes=output_dim, aux_logits=False)
        #del self.fc
        self.Conv2d_1a_3x3 = torchvision.models.inception.BasicConv2d(1, 32, kernel_size=3, stride=2)
        self.final_feat_dim = 2048
        self.preprocess = torchvision.transforms.Compose([
            torchvision.transforms.Resize((299, 299)),
        ])
       

    def load_sl_official_weights(self, progress=True):
        state_dict = load_state_dict_from_url(torchvision.models.inception.Inception_V3_Weights.IMAGENET1K_V1.url,
                                              progress=progress)

        del state_dict['Conv2d_1a_3x3.weight']
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        # if len(missing) > 0:
            # raise AssertionError('Model code may be incorrect')

    def load_ssl_official_weights(self, progress=True):
        raise NotImplemented

    def _forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(1)
        x = self.preprocess(x)
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.maxpool1(x)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self.maxpool2(x)
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        # N x 768 x 17 x 17
        aux: Optional[Tensor] = None
        if self.AuxLogits is not None:
            if self.training:
                aux = self.AuxLogits(x)
        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = self.avgpool(x)
        # N x 2048 x 1 x 1
        x = self.dropout(x)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 2048
        x = self.fc(x)
        # N x 1000 (num_classes)
        return x, aux
    
class LSTMModel1(nn.Module):
    def __init__(self, input_size, pooling_hidden, p_dropout, output_dim, **kwargs):
        super(LSTMModel1, self).__init__()
        
        self.batch_norm1 = nn.BatchNorm1d(input_size)
        self.lstm1 = nn.LSTM(input_size, pooling_hidden, batch_first=True)
        
        self.batch_norm2 = nn.BatchNorm1d(pooling_hidden)
        self.lstm2 = nn.LSTM(pooling_hidden, pooling_hidden, batch_first=True)
        
        #self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8, batch_first=True)

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p_dropout)
        self.fc = nn.Linear(pooling_hidden, output_dim)

    def forward(self, x, lengths=None):
        # x -> [10, 13, 125]
        x = x.permute(0, 2, 1)
        x = self.batch_norm1(x.transpose(1, 2)).transpose(1, 2)
        x, _ = self.lstm1(x)
        
        x = self.batch_norm2(x.transpose(1, 2)).transpose(1, 2)
        x, _ = self.lstm2(x)
        
        # attn_output, _ = self.attention(x, x, x)
        # x = torch.mean(attn_output, dim=1)

        x = self.flatten(x[:, -1, :])
        x = self.dropout(x)
        x = self.fc(x)
        return x

class CNNClassifier(nn.Module):
    def __init__(self, dummy, output_dim, **kwargs):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        self.fc = nn.Linear(32, output_dim)

    def forward(self, x, lengths=None):
        # x shape: [B, Feat_Dim, T]
        x = x.unsqueeze(1)  # shape becomes [B, 1, Feat_Dim, T]
        x = F.relu(self.bn1(self.conv1(x)))  # [B, 16, Feat_Dim, T]
        x = F.relu(self.bn2(self.conv2(x)))  # [B, 32, Feat_Dim, T]
        x = self.pool(x)  # [B, 32, 1, 1]
        x = x.view(x.size(0), -1)  # Flatten to [B, 32]
        x = self.fc(x)  # [B, num_classes]
        return x

##### VIT
class ViT(nn.Module):
    def __init__(self, dummmy, output_dim, image_size=256, patch_size=32, dim=1024, depth=6, heads=8, mlp_dim=2048, pool = 'cls', channels = 1, dim_head = 64, dropout = 0., emb_dropout = 0., **kwargs):
        super().__init__()
        image_height, image_width = commons.pair(image_size)
        patch_height, patch_width = commons.pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.pretrain = False
        self.preprocess = torchvision.transforms.Compose([
            torchvision.transforms.Resize((256,256))
        ])

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = modules.TransformerVIT(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, output_dim)

    def forward(self, img):
        if self.pretrain == False:
            img = img.unsqueeze(1)
            img = self.preprocess(img)

        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

class SimMIM(nn.Module):
    def __init__(
        self,
        *,
        encoder,
        masking_ratio = 0.5
    ):
        super().__init__()
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)

        self.encoder = encoder
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]

        self.to_patch = encoder.to_patch_embedding[0]
        self.patch_to_emb = nn.Sequential(*encoder.to_patch_embedding[1:])

        pixel_values_per_patch = encoder.to_patch_embedding[2].weight.shape[-1]

        # simple linear head

        self.mask_token = nn.Parameter(torch.randn(encoder_dim))
        self.to_pixels = nn.Linear(encoder_dim, pixel_values_per_patch)

    def forward(self, img):
        device = img.device

        # get patches

        patches = self.to_patch(img)
        batch, num_patches, *_ = patches.shape

        # for indexing purposes

        batch_range = torch.arange(batch, device = device)[:, None]

        # get positions

        pos_emb = self.encoder.pos_embedding[:, 1:(num_patches + 1)]

        # patch to encoder tokens and add positions

        tokens = self.patch_to_emb(patches)
        tokens = tokens + pos_emb

        # prepare mask tokens

        mask_tokens = repeat(self.mask_token, 'd -> b n d', b = batch, n = num_patches)
        mask_tokens = mask_tokens + pos_emb

        # calculate of patches needed to be masked, and get positions (indices) to be masked

        num_masked = int(self.masking_ratio * num_patches)
        masked_indices = torch.rand(batch, num_patches, device = device).topk(k = num_masked, dim = -1).indices
        masked_bool_mask = torch.zeros((batch, num_patches), device = device).scatter_(-1, masked_indices, 1).bool()

        # mask tokens

        tokens = torch.where(masked_bool_mask[..., None], mask_tokens, tokens)

        # attend with vision transformer

        encoded = self.encoder.transformer(tokens)

        # get the masked tokens

        encoded_mask_tokens = encoded[batch_range, masked_indices]

        # small linear projection for predicted pixel values

        pred_pixel_values = self.to_pixels(encoded_mask_tokens)

        # get the masked patches for the final reconstruction loss

        masked_patches = patches[batch_range, masked_indices]

        # calculate reconstruction loss

        recon_loss = F.l1_loss(pred_pixel_values, masked_patches) / num_masked
        return recon_loss

class SimpleMaskedModel(nn.Module):
    def __init__(self, dummy, output_dim, **kwargs):
        super(SimpleMaskedModel, self).__init__()

        self.preprocess = torchvision.transforms.Compose([
            torchvision.transforms.Resize((256,256))
        ])

        self.v = ViT(
            image_size = 256,
            patch_size = 32,
            num_classes = output_dim,
            dim = 1024,
            depth = 6,
            heads = 8,
            mlp_dim = 2048,
            channels=1
        )

        self.mim = SimMIM(
            encoder = self.v,
            masking_ratio = 0.5  # they found 50% to yield the best results
        )

    def forward(self, x):
        # x shape: [128, 80, 32]
        x = x.unsqueeze(1)
        x = self.preprocess(x)
        loss = self.mim(x)
        return loss

#################### REG SSL ###########################
class HeadCatPrediction(nn.Module):
    def __init__(self, pooling_hidden, regress_hidden_dim, regress_dropout, regress_layers, output_dim, **kwargs):
        super(HeadCatPrediction, self).__init__()

        self.inp_drop = nn.Dropout(regress_dropout)
        self.fc=nn.ModuleList([nn.Sequential(
                nn.Linear(pooling_hidden, regress_hidden_dim), 
                nn.LayerNorm(regress_hidden_dim), nn.ReLU(), nn.Dropout(regress_dropout))])

        for lidx in range(regress_layers-1):
            self.fc.append(nn.Sequential(
                    nn.Linear(regress_hidden_dim, regress_hidden_dim), 
                    nn.LayerNorm(regress_hidden_dim), nn.ReLU(), nn.Dropout(regress_dropout)))

        self.out = nn.Sequential(nn.Linear(regress_hidden_dim, output_dim))

        self.dense = nn.Linear(pooling_hidden, regress_hidden_dim)
        self.dropout = nn.Dropout(regress_dropout)
        self.out_proj = nn.Linear(regress_hidden_dim, output_dim)

    # def get_repr(self, x):
    #     h = self.inp_drop(x)
    #     for lidx, fc in enumerate(self.fc):
    #         h=fc(h)
    #     return h

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        out_dim = self.out_proj(x)
        # h = self.get_repr(x)
        # out_dim = self.out(h)
        return out_dim