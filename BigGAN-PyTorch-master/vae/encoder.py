import numpy as np
import math
import functools

import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P
from torchvision.models.resnet import Bottleneck, ResNet
import layers


class Encoder_inv(ResNet):
    def __init__(self, latent_dim):
        super(Encoder_inv, self).__init__(Bottleneck, [3, 4, 6, 4])
        self.downsample = nn.Sequential(
            nn.Conv2d(512 * 4, latent_dim, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(latent_dim))
        self.out_layer = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        :param x: [NCHW]
        :return:
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = self.downsample(x)
        x = torch.flatten(x, 1)
        x = self.out_layer(x)
        return x


class Encoder_rd(ResNet):
    def __init__(self, latent_dim):
        super(Encoder_rd, self).__init__(Bottleneck, [3, 4, 6, 4])
        self.resblock1 = ResBlock(512 * 4, latent_dim * 2)
        self.resblock2 = ResBlock(latent_dim * 2, latent_dim * 2)

    def forward(self, x):
        """
        :param x: [NCHW]
        :return:
        """
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)  # [batch_size, 2048, 1, 1]

        x = self.resblock1(x)
        x = self.resblock2(x)  # [batch_size, latent_dim *2]
        mean, var = torch.split(x, x.shape[1] // 2, 1)
        epsilon = torch.rand(mean.shape)
        x = var * epsilon + mean
        return x


class ResBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(ResBlock, self).__init__()
        if inplanes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, planes, 1),
                nn.BatchNorm2d(planes))
        else:
            self.shortcut = nn.Identity()
        self.res_layer = nn.Sequential(
            nn.Conv2d(inplanes, planes, 1),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, 1),
            nn.BatchNorm2d(planes)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.res_layer(x)
        x += identity
        x = self.relu(x)
        return x


def Encoder(arch='ResBlock_inv', latent_dim=120, labeled_percent=0.1):
    encoder_dict = {"ResBlock_inv": Encoder_inv, "ResBlock_rand": Encoder_rd}
    assert arch in encoder_dict.keys()
    return encoder_dict[arch](latent_dim)
