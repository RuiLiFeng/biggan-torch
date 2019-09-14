import torch
import torch.nn.functional as F
from vae.vgg_utils import load_vgg_from_local


def loss_reconstruction(reals, fakes, loss_coef):
    vgg = load_vgg_from_local()
    reals = vgg(reals)
    fakes = vgg(fakes)
    loss = loss_coef * torch.mean(reals - fakes)
    return loss


def loss_tie_g(inv_z, encoder_z, lam=10):
    loss = torch.mean(encoder_z - inv_z)
    return loss


def loss_tie_d(inv_z, encoder_z, lam=10):
    pass
