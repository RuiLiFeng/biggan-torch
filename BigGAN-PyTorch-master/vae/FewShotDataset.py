import os
import os.path
import sys
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm, trange

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.datasets.utils import download_url, check_integrity
import torch.utils.data as data
from datasets import ILSVRC_HDF5
from torch.utils.data import DataLoader


class Subset(data.dataset):
    """
    Subset of a dataset at specified indices.
    """
    def __init__(self, dataset: ILSVRC_HDF5, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]

    def __len__(self):
        return len(self.indices)


def generate_keep_table(labels, keep_prop):
    keep_table = []
    remove_table = []
    num_imgs = len(labels)
    start = 0
    for i in range(num_imgs):
        if labels[i] != labels[start]:
            class_length = i - start
            perm = torch.randperm(class_length) + start
            keep_table.append(perm[:int(keep_prop * class_length)])
            remove_table.append(perm[int(keep_prop * class_length):])
    keep_table = torch.cat(keep_table)
    remove_table = torch.cat(remove_table)
    return keep_table, remove_table


def generate_fewshot_dset(dataset: ILSVRC_HDF5, keep_prop):
    labels = dataset.labels[:]
    keep_table, remove_table = generate_keep_table(labels, keep_prop)
    labeled_dset = Subset(dataset, keep_table)
    unlabeled_dset = Subset(dataset, remove_table)
    return labeled_dset, unlabeled_dset
