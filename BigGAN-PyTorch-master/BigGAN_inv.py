import numpy as np
import math
import functools

import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P
import BigGAN
import invert
import layers
from sync_batchnorm import SynchronizedBatchNorm2d as SyncBatchNorm2d


class Generator(BigGAN.Generator):
    def __init__(self, **kwargs):
        super(Generator, self).__init__(**kwargs)
        self.invert = invert.Invert(z_dim=self.dim_z, depth=8)

    def forward(self, z, y):
        # Apply invertible network
        z = self.invert(z)
        return super(Generator, self).forward(z, y)

    def load_state_dict(self, state_dict, strict=True):
        # Load state for those in state_dict, remain others unchanged.
        model_dict = self.state_dict()
        # Check if the network structure is changed.
        new_dict = {k: v for k, v in state_dict.items() if k in model_dict.keys()}
        model_dict.update(new_dict)
        super(Generator, self).load_state_dict(model_dict)


class Discriminator(BigGAN.Discriminator):
    def __init__(self, unconditional=False, **kwargs):
        super(Discriminator, self).__init__(**kwargs)
        self.unconditional = unconditional

    def forward(self, x, y=None):
        if self.unconditional:
            # Stick x into h for cleaner for loops without flow control
            h = x
            # Loop over blocks
            for index, blocklist in enumerate(self.blocks):
                for block in blocklist:
                    h = block(h)
            # Apply global sum pooling as in SN-GAN
            h = torch.sum(self.activation(h), [2, 3])
            # Get initial class-unconditional output
            out = self.linear(h)
            # Get projection of final featureset onto class vectors and add to evidence
            out = out
            return out
        else:
            return super(Discriminator, self).forward(x, y)


class G_D(BigGAN.G_D):
    def __init__(self, G, D):
        super(G_D, self).__init__(G, D)

