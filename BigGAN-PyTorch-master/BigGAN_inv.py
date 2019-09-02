import numpy as np
import math
import functools

import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P
import BigGANdeep
import invert
import layers
from sync_batchnorm import SynchronizedBatchNorm2d as SyncBatchNorm2d


class Generator_inv(BigGANdeep.Generator):
    def __init__(self, hidden, **kwargs):
        super(Generator_inv, self).__init__(**kwargs)
        self.invert = invert.Invert(z_dim=self.dim_z, depth=8, hidden=hidden)

    def forward(self, z, y):
        # Apply invertible network
        z = self.invert(z)
        # If hierarchical, concatenate zs and ys
        if self.hier:
            z = torch.cat([y, z], 1)
            y = z
        # First linear layer
        h = self.linear(z)
        # Reshape
        h = h.view(h.size(0), -1, self.bottom_width, self.bottom_width)
        # Loop over blocks
        for index, blocklist in enumerate(self.blocks):
            # Second inner loop in case block has multiple layers
            for block in blocklist:
                h = block(h, y)

        # Apply batchnorm-relu-conv-tanh at output
        return torch.tanh(self.output_layer(h))


class Discriminator(BigGANdeep.Discriminator):
    def __init__(self, **kwargs):
        super(Discriminator, self).__init__(kwargs)


class G_D(BigGANdeep.G_D):
    def __init__(self, G, D):
        super(G_D, self).__init__(G, D)
