import os
import sys
from argparse import ArgumentParser
import h5py as h5
sys.path.append('..')
from utils import get_data_loaders

import numpy as np


def prepare_parser():
    usage = 'Parser for ImageNet HDF5 scripts.'
    parser = ArgumentParser(description=usage)
    parser.add_argument(
        '--dataset', type=str, default='ILSVRC128.hdf5',
        help='Which Dataset to train on, out of I128, I256, C10, C100;'
             'Append "_hdf5" to use the hdf5 version for ISLVRC (default: %(default)s)')
    parser.add_argument(
        '--data_root', type=str, default='/gpub/temp/imagenet2012/hdf5',
        help='Default location where data is stored (default: %(default)s)')
    parser.add_argument(
        '--chunk_size', type=int, default=500,
        help='Default overall batchsize (default: %(default)s)')
    parser.add_argument(
        '--compression', action='store_true', default=False,
        help='Use LZF compression? (default: %(default)s)')
    parser.add_argument(
        '--result_dir', type=str, default='/gpub/temp/imagenet2012/hdf5',
        help='Default directory to store the result hdf5 file (default: %(default)s)')
    parser.add_argument(
        '--keep_prop', type=float, default=0.01,
        help='Default keep prop for labeled data in the whole dataset (default: %(default)s)')
    parser.add_argument(
        '--num_workers', type=int, default=16,
        help='Defualt num_works for DataLoader (default: %(default)s)')
    parser.add_argument(
        '--batch_size', type=int, default=1,
        help='Default batch size for DataLoader (default: %(default)s)')
    return parser


def run(config):
    # Update compression entry
    config['compression'] = 'lzf' if config['compression'] else None  # No compression; can also use 'lzf'

    # Get datasets:
    kwargs = {'num_workers': config['num_workers'], 'pin_memory': False, 'drop_last': False}
    FullDset = get_data_loaders(dataset=config['dataset'],
                                batch_size=config['batch_size'],
                                shuffle=False,
                                data_root=config['data_root'],
                                use_multiepoch_sampler=False,
                                result_dir=config["result_dir"],
                                **kwargs)[0]
    img_shape = FullDset.dataset['imgs'][0].shape
    label_shape = FullDset.dataset['labels'][0].shape
    print('Start generating keep table with keep prop %f...' % config['keep_prop'])
    KeepTable, RemoveTable = generate_keep_table(FullDset['labels'][:], config['keep_prop'])

    print('Starting to convert %s into an departed HDF5 file with keep porp %f, chunk size %i and compression %s...'
          % (config['keep_prop'], config['dataset'], config['chunk_size'], config['compression']))
    # Create datasets
    with h5.File(config['result_dir'] + '/ILSVRC128_%d.hdf5'.format(config['keep_prop']), 'w') as f:
        print("Producing dataset of labeled data %d, unlabeled data %d"
              % (int(config['keep_prop'] * len(FullDset['labels'])),
                 int((1 - config['keep_prop']) * len(FullDset['labels']))))
        limgs_dset = f.create_dataset('limgs', (1,) + img_shape, dtype='uint8',
                                      maxshape=((len(KeepTable),) + img_shape),
                                      chunks=((config['chunk_size'],) + img_shape),
                                      compression=config['compression'])
        llabels_dset = f.create_dataset('llabels', (1,) + label_shape, dtype='int64',
                                        maxshape=((len(KeepTable),) + label_shape),
                                        chunks=((config['chunk_size'],) + label_shape),
                                        compression=config['compression'])
        print("Labeled dataset with chunk size %i for imgs, %i for labels" %
              (limgs_dset.chunks, llabels_dset.chunks))
        uimgs_dset = f.create_dataset('uimgs', (1,) + img_shape, dtype='uint8',
                                      maxshape=((len(RemoveTable),) + img_shape),
                                      chunks=((config['chunk_size'],) + img_shape),
                                      compression=config['compression'])
        ulabels_dset = f.create_dataset('ulabels', (1,) + label_shape, dtype='int64',
                                        maxshape=((len(RemoveTable),) + label_shape),
                                        chunks=((config['chunk_size'],) + label_shape),
                                        compression=config['compression'])
        print("Unlabeled dataset with chunk size %i for imgs, %i for labels" %
              (uimgs_dset.chunks, ulabels_dset.chunks))
    for index in KeepTable:
        x = FullDset['imgs'][index]
        y = FullDset['labels'][index]
        x = np.expand_dims(x, 0)
        y = np.expand_dims(y, 0)
        with h5.File(config['result_dir'] + '/ILSVRC128_%d.hdf5'.format(config['keep_prop']), 'a') as f:
            if index == 0:
                f['limgs'][...] = x
                f['llabels'][...] = y
            else:
                f['limgs'].resize(f['limgs'].shape[0] + x.shape[0], axis=0)
                f['limgs'][-x.shape[0]:] = x
                f['llabels'].resize(f['llabels'].shape[0] + y.shape[0], axis=0)
                f['llabels'][-y.shape[0]:] = y
    for index in RemoveTable:
        x = FullDset['imgs'][index]
        y = FullDset['labels'][index]
        x = np.expand_dims(x, 0)
        y = np.expand_dims(y, 0)
        with h5.File(config['result_dir'] + '/ILSVRC128_%d.hdf5'.format(config['keep_prop']), 'a') as f:
            if index == 0:
                f['uimgs'][...] = x
                f['ulabels'][...] = y
            else:
                f['uimgs'].resize(f['uimgs'].shape[0] + x.shape[0], axis=0)
                f['uimgs'][-x.shape[0]:] = x
                f['ulabels'].resize(f['ulabels'].shape[0] + y.shape[0], axis=0)
                f['ulabels'][-y.shape[0]:] = y
    FullDset.close()


def generate_keep_table(labels, keep_prop):
    keep_table = []
    remove_table = []
    num_imgs = len(labels)
    start = 0
    for i in range(num_imgs):
        if labels[i] != labels[start]:
            indexs = list(range(start, i+1))
            keep = np.random.choice(indexs, size=int(keep_prop * len(indexs)))
            keep_table.append(keep)
            remove_table.append([index for index in indexs if index not in keep])
    keep_table = np.concatenate(keep_table)
    remove_table = np.concatenate(remove_table)
    return keep_table, remove_table


def main():
    # parse command line and run
    parser = prepare_parser()
    config = vars(parser.parse_args())
    print(config)
    run(config)


if __name__ == '__main__':
    main()
