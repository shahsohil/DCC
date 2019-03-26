from __future__ import print_function
import os
import random
import numpy as np
import argparse
import cPickle
from config import cfg, get_data_dir, get_output_dir
import data_params as dp

import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from extractSDAE import extract_sdae_mnist, extract_sdae_reuters, extract_sdae_ytf, extract_sdae_coil100, extract_sdae_yale, extract_sdae_easy
from extractconvSDAE import extract_convsdae_mnist, extract_convsdae_coil100, extract_convsdae_ytf, extract_convsdae_yale
from custom_data import DCCPT_data

# Parse all the input argument
parser = argparse.ArgumentParser(description='Module for extracting features from pretrained SDAE')
parser.add_argument('--manualSeed', default=cfg.RNG_SEED, type=int, help='manual seed')
parser.add_argument('--data', dest='db', type=str, default='mnist', help='name of the dataset')
parser.add_argument('--net', dest='torchmodel', help='path to the weights file', default=None, type=str)
parser.add_argument('--features', dest='feat', help='path to the feature file', default=None, type=str)
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--deviceID', type=int, help='deviceID', default=0)
parser.add_argument('--dim', type=int, help='dimension', default=10)
parser.add_argument('--h5', dest='h5', help='to store as h5py file', default=False, type=bool)

def main():
    global args

    args = parser.parse_args()
    datadir = get_data_dir(args.db)
    outputdir = get_output_dir(args.db)

    use_cuda = torch.cuda.is_available()

    # Set the seed for reproducing the results
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.manualSeed)
        torch.backends.cudnn.enabled = True
        cudnn.benchmark = True

    reluslope = 0.0
    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}
    trainset = DCCPT_data(root=datadir, train=True, h5=args.h5)
    testset = DCCPT_data(root=datadir, train=False, h5=args.h5)
    if args.db == 'mnist':
        net = extract_sdae_mnist(slope=reluslope, dim=args.dim)
    elif args.db == 'reuters' or args.db == 'reuters10k' or args.db == 'rcv1':
        net = extract_sdae_reuters(slope=reluslope, dim=args.dim)
    elif args.db == 'ytf':
        net = extract_sdae_ytf(slope=reluslope, dim=args.dim)
    elif args.db == 'coil100':
        net = extract_sdae_coil100(slope=reluslope, dim=args.dim)
    elif args.db == 'yale':
        net = extract_sdae_yale(slope=reluslope, dim=args.dim)
    elif args.db == 'cmnist':
        net = extract_convsdae_mnist(slope=reluslope)
    elif args.db == 'ccoil100':
        net = extract_convsdae_coil100(slope=reluslope)
    elif args.db == 'cytf':
        net = extract_convsdae_ytf(slope=reluslope)
    elif args.db == 'cyale':
        net = extract_convsdae_yale(slope=reluslope)
    elif args.db == dp.easy.name:
        net = extract_sdae_easy(slope=reluslope, dim=args.dim)
    else:
        raise ValueError("Unexpected database %s" % args.db)

    totalset = torch.utils.data.ConcatDataset([trainset, testset])
    dataloader = torch.utils.data.DataLoader(totalset, batch_size=100, shuffle=False, **kwargs)

    # copying model params from checkpoint
    filename = os.path.join(outputdir, args.torchmodel)
    if os.path.isfile(filename):
        print("==> loading params from checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        net.load_state_dict(checkpoint['state_dict'])
    else:
        print("==> no checkpoint found at '{}'".format(filename))
        raise

    if use_cuda:
        net.cuda()

    print('Extracting features ...')
    features, features_dr, labels = extract(dataloader, net, use_cuda)
    print('Done.\n')

    feat_path = os.path.join(datadir, args.feat)
    if args.h5:
        import h5py
        fo = h5py.File(feat_path + '.h5', 'w')
        fo.create_dataset('labels', data=labels)
        fo.create_dataset('Z', data=np.squeeze(features_dr))
        fo.create_dataset('data', data=np.squeeze(features))
        fo.close()
    else:
        fo = open(feat_path + '.pkl', 'wb')
        cPickle.dump({'labels': labels, 'Z': np.squeeze(features_dr), 'data': np.squeeze(features)}, fo, protocol=2)
        fo.close()

def extract(dataloader, net, use_cuda):
    net.eval()

    original = []
    features = []
    labels = []

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        inputs_Var = Variable(inputs, volatile=True)
        enc, dec = net(inputs_Var)
        features += list(enc.data.cpu().numpy())
        labels += list(targets)
        original += list(inputs.cpu().numpy())

    original = np.asarray(original).astype(np.float32)
    if len(original.shape) != len(inputs.shape):
        original = original[:,np.newaxis,:,:]

    return original, np.asarray(features).astype(np.float32), np.asarray(labels)

if __name__ == '__main__':
    main()