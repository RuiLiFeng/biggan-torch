#!/bin/bash
python make_hdf5.py --dataset I128 --batch_size 256 --data_root "/gpub/imagenet_raw"
python calculate_inception_moments.py --dataset I128_hdf5 --data_root "/gpub/temp/imagenet2012/hdf5"