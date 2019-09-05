#!/bin/bash
python /ghome/fengrl/home/biggan/torch-imp/biggan-torch/BigGAN-PyTorch-master/make_hdf5.py --dataset I128 --batch_size 64 --data_root "/gpub/imagenet_raw"
python /ghome/fengrl/home/biggan/torch-imp/biggan-torch/BigGAN-PyTorch-master/calculate_inception_moments.py --dataset I128_hdf5 --data_root "/gpub/temp/imagenet2012/hdf5"