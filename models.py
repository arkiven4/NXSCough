import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url

import torchvision

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
    
class ResNet18(torchvision.models.resnet.ResNet):
    def __init__(self, num_classes, track_bn=True):
        def norm_layer(*args, **kwargs):
            return nn.BatchNorm2d(*args, **kwargs, track_running_stats=track_bn)
        super().__init__(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], norm_layer=norm_layer, num_classes=num_classes)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.final_feat_dim = 512

    def load_sl_official_weights(self, progress=True):
        state_dict = load_state_dict_from_url(torchvision.models.resnet.ResNet18_Weights.IMAGENET1K_V1.url,
                                              progress=progress)

        del state_dict['conv1.weight']
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        # if len(missing) > 0:
            # raise AssertionError('Model code may be incorrect')

    def load_ssl_official_weights(self, progress=True):
        raise NotImplemented

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = x.unsqueeze(1)
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

class ResNet101(torchvision.models.resnet.ResNet):
    def __init__(self, num_classes, track_bn=True):
        def norm_layer(*args, **kwargs):
            return nn.BatchNorm2d(*args, **kwargs, track_running_stats=track_bn)
        super().__init__(torchvision.models.resnet.Bottleneck, [3, 4, 23, 3], norm_layer=norm_layer, num_classes=num_classes)
        #del self.fc
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.final_feat_dim = 2048

    def load_sl_official_weights(self, progress=True):
        state_dict = load_state_dict_from_url(torchvision.models.resnet.ResNet101_Weights.IMAGENET1K_V2.url,
                                              progress=progress)

        del state_dict['conv1.weight']
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        # if len(missing) > 0:
            # raise AssertionError('Model code may be incorrect')

    def load_ssl_official_weights(self, progress=True):
        raise NotImplemented

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = x.unsqueeze(1)
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