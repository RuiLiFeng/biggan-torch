import numpy as np
import math
import functools

import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P

import layers


class Encoder(nn.Module):
    def __init__(self,
                 latent_dim,
                 resolution=128,
                 arch='ResNet50'):
        super(Encoder, self).__init__()
        self._latent_dim = latent_dim
        self._resolution = resolution
        self._arch = arch