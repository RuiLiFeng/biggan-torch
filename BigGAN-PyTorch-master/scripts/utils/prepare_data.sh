#!/bin/bash
python /ghome/fengrl/home/biggan/torch-imp/biggan-torch/BigGAN-PyTorch-master/make_hdf5.py \
--dataset I128 --batch_size 8 --data_root "/gpub/temp/imagenet2012/downloads/extracted"
python /ghome/fengrl/home/biggan/torch-imp/biggan-torch/BigGAN-PyTorch-master/calculate_inception_moments.py \
--dataset I128_hdf5 --batch_size 8