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


def get_weight(shape, gain=np.sqrt(2), use_wscale=False, lrmul=1.0):
    """
    get_weight for dense layers
    :param shape: shape of weight, [in_channels, hidden]
    :param gain:
    :param use_wscale:
    :param lrmul:
    :return:
    """
    fan_in = np.prod(shape[:-1])  # [kernel, kernel, fmaps_in, fmaps_out] or [in, out]
    he_std = gain / np.sqrt(fan_in)  # He init

    # Equalized learning rate and custom learning rate multiplier.
    if use_wscale:
        init_std = 1.0 / lrmul
        runtime_coef = he_std * lrmul
    else:
        init_std = he_std / lrmul
        runtime_coef = lrmul

    # Create variable.
    return P(torch.normal(0, init_std * torch.ones(shape))) * runtime_coef


def get_bias(in_channel, lrmul=1.0):
    """
    get_bias for dense layer.
    :param in_channel: Must be list, shape of bias
    :param lrmul:
    :return:
    """
    if len(in_channel) > 1:
        in_channel = np.prod([d.value for d in in_channel])
    return P(torch.zeros(in_channel)) * lrmul


class Dense(nn.Module):
    def __init__(self, in_channels, hidden,
                 gain=np.sqrt(2), use_wscale=True, mul_lrmul=0.01, bias_lrmul=0.01):
        """
        :param in_channels: feature shapes, batch_size is not contained, must be int.
        :param hidden: Dimension of hidden layers.
        """
        super(Dense, self).__init__()
        # Used to check input in forward()
        self._in_channels = in_channels

        self.w = get_weight([in_channels, hidden], gain=gain, use_wscale=use_wscale, lrmul=mul_lrmul)
        self.bias = get_bias([in_channels], bias_lrmul)

    def forward(self, x):
        assert len(x.shape) == 2
        assert x.shape[1] == self._in_channels
        if len(x.shape) > 2:
            x = x.reshape([-1, np.prod([d.value for d in x.shape[1:]])])
        self.w = self.w.type(x.dtype)
        self.bias = self.bias.type(x.dtype)
        return torch.matmul(x, self.w) + self.bias


class phi(nn.Module):
    def __init__(self, in_channels, hidden, out_channels=None, alpha=0.2, **kwargs):
        super(phi, self).__init__()
        self._in_channels = in_channels
        self._out_channels = in_channels if out_channels is None else out_channels
        self.dense1 = Dense(in_channels, hidden, **kwargs)
        self.act1 = torch.nn.LeakyReLU(alpha)
        self.dense2 = Dense(hidden, out_channels, **kwargs)
        self.act2 = torch.nn.LeakyReLU(alpha)

    def forward(self, x):
        x = self.dense1(x)
        x = self.act1(x)
        x = self.dense2(x)
        x = self.act2(x)
        return x


class step(nn.Module):
    def __init__(self, in_channels, hidden, is_reverse, **kwargs):
        super(step, self).__init__()
        self._hidden = hidden
        self._in_channels = in_channels
        self._is_reverse = is_reverse
        self.phi = phi(in_channels, hidden, in_channels // 2, **kwargs)

    def _reverse(self, h):
        return h[:, :: -1]

    def forward(self, x):
        assert x.shape[1:] == self._in_channels
        if len(x.shape) > 2:
            x.reshape([-1, np.prod(x.shape[1:])])
        x = self._reverse(x)
        x1 = x[:, :x.shape[1] // 2]
        x2 = x[:, x.shape[1] // 2:]
        x2 += self.phi(x1)
        x = torch.cat((x1, x2), dim=1)
        return x


class Invert(nn.Module):
    def __init__(self, z_dim, depth, hidden, is_reverse=True):
        """
        Invertible network.
        :param z_dim: Must be int.
        :param depth: How many coupling layer are used.
        :param hidden:
        :param is_reverse:
        """
        super(Invert, self).__init__()
        self._z_dim = z_dim
        self._depth = depth
        self._is_reverse = is_reverse
        self.steps = []
        for i in range(self._depth):
            self.steps.append(step(self._z_dim, hidden, is_reverse))

    def forward(self, x):
        for stp in self.steps:
            x = stp(x)
        return x

