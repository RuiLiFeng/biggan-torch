import numpy as np
import math
import functools

import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P
from torchvision.models import resnet as R
import layers


class Encoder(R.ResNet):
    def __init__(self,
                 latent_dim,
                 resolution=128,
                 arch='ResNet50',
                 pretrained=False,
                 **kwargs):
        self._latent_dim = latent_dim
        self._resolution = resolution
        self._arch = arch
        super(Encoder, self).__init__(R.Bottleneck, [3, 4, 6, 3], **kwargs)
        if pretrained:
            state_dict = {}
            self.load_state_dict(state_dict)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)  # [batch_size, 512 * 4]
        return x


class FcRes(nn.Module):
    def __init__(self, in_channel, out_channel, depth, bn=False):
        super(FcRes, self).__init__()
        self._in_channel = in_channel
        self._out_channel = out_channel
        self._depth = depth
        if bn:
            norm_layer = nn.BatchNorm2d

