from __future__ import print_function
import os
import random
import math
import numpy as np
import scipy.io as sio
import argparse
from config import cfg, get_data_dir, get_output_dir, AverageMeter

import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from extractSDAE import extract_sdae_mnist, extract_sdae_reuters, extract_sdae_ytf, extract_sdae_coil100, extract_sdae_yale
from extractconvSDAE import extract_convsdae_mnist, extract_convsdae_coil100, extract_convsdae_ytf, extract_convsdae_yale
from custom_data import DCCPT_data, DCCFT_data, DCCSampler
from DCCLoss import DCCWeightedELoss, DCCLoss
from DCCComputation import makeDCCinp, computeHyperParams, computeObj

# used for logging to TensorBoard
from tensorboard_logger import configure, log_value

# Parse all the input argument
parser = argparse.ArgumentParser(description='PyTorch DCC Finetuning')
parser.add_argument('--data', dest='db', type=str, default='mnist',
                    help='Name of the dataset. The name should match with the output folder name.')
parser.add_argument('--batchsize', type=int, default=cfg.PAIRS_PER_BATCH, help='batch size used for Finetuning')
parser.add_argument('--nepoch', type=int, default=500, help='maximum number of iterations used for Finetuning')
# By default M = 20 is used. For convolutional SDAE M=10 was used.
# Similarly, for different NW architecture different value for M may be required.
parser.add_argument('--M', type=int, default=20, help='inner number of epochs at which to change lambda')
parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--manualSeed', default=cfg.RNG_SEED, type=int, help='manual seed')
parser.add_argument('--net', dest='torchmodel', help='path to the pretrained weights file', default=None, type=str)
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--level', default=0, type=int, help='epoch to resume from')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--deviceID', type=int, help='deviceID', default=0)
parser.add_argument('--h5', dest='h5', action='store_true', help='to store as h5py file')
parser.add_argument('--dim', type=int, help='dimension of embedding space', default=10)
parser.add_argument('--tensorboard', help='Log progress to TensorBoard', action='store_true')
parser.add_argument('--id', type=int, help='identifying number for storing tensorboard logs')


def main():
    global args, oldassignment

    args = parser.parse_args()
    datadir = get_data_dir(args.db)
    outputdir = get_output_dir(args.db)

    if args.tensorboard:
        # One should create folder for storing logs
        loggin_dir = os.path.join(outputdir, 'runs', 'DCC')
        if not os.path.exists(loggin_dir):
            os.makedirs(loggin_dir)
        configure(os.path.join(loggin_dir, '%s' % (args.id)))

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
    startepoch = 0
    kwargs = {'num_workers': 5, 'pin_memory': True} if use_cuda else {}

    # setting up dataset specific objects
    trainset = DCCPT_data(root=datadir, train=True, h5=args.h5)
    testset = DCCPT_data(root=datadir, train=False, h5=args.h5)
    numeval = len(trainset) + len(testset)

    # extracting training data from the pretrained.mat file
    data, labels, pairs, Z, sampweight = makeDCCinp(args)

    # For simplicity, I have created placeholder for each datasets and model
    if args.db == 'mnist':
        net = extract_sdae_mnist(slope=reluslope, dim=args.dim)
    elif args.db == 'reuters' or args.db == 'rcv1':
        net = extract_sdae_reuters(slope=reluslope, dim=args.dim)
    elif args.db == 'ytf':
        net = extract_sdae_ytf(slope=reluslope, dim=args.dim)
    elif args.db == 'coil100':
        net = extract_sdae_coil100(slope=reluslope, dim=args.dim)
    elif args.db == 'yale':
        net = extract_sdae_yale(slope=reluslope, dim=args.dim)
    elif args.db == 'cmnist':
        net = extract_convsdae_mnist(slope=reluslope)
        data = data.reshape((-1,1,28,28))
    elif args.db == 'ccoil100':
        net = extract_convsdae_coil100(slope=reluslope)
        data = data.reshape((-1,3,128,128))
    elif args.db == 'cytf':
        net = extract_convsdae_ytf(slope=reluslope)
        data = data.reshape((-1,3,55,55))
    elif args.db == 'cyale':
        net = extract_convsdae_yale(slope=reluslope)
        data = data.reshape((-1,1,168,192))

    totalset = torch.utils.data.ConcatDataset([trainset, testset])

    # computing and initializing the hyperparams
    _sigma1, _sigma2, _lambda, _delta, _delta1, _delta2, lmdb, lmdb_data = computeHyperParams(pairs, Z)
    oldassignment = np.zeros(len(pairs))
    stopping_threshold = int(math.ceil(cfg.STOPPING_CRITERION * float(len(pairs))))

    # Create dataset and random batch sampler for Finetuning stage
    trainset = DCCFT_data(pairs, data, sampweight)
    batch_sampler = DCCSampler(trainset, shuffle=True, batch_size=args.batchsize)

    # copying model params from Pretrained (SDAE) weights file
    filename = os.path.join(outputdir, args.torchmodel)
    if os.path.isfile(filename):
        print("==> loading params from checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        net.load_state_dict(checkpoint['state_dict'])
    else:
        print("==> no checkpoint found at '{}'".format(filename))
        raise

    # creating objects for loss functions, U's are initialized to Z here
    # Criterion1 corresponds to reconstruction loss
    criterion1 = DCCWeightedELoss(size_average=True)
    # Criterion2 corresponds to sum of pairwise and data loss terms
    criterion2 = DCCLoss(Z.shape[0], Z.shape[1], Z, size_average=True)

    if use_cuda:
        net.cuda()
        criterion1 = criterion1.cuda()
        criterion2 = criterion2.cuda()

    # setting up data loader for training and testing phase
    trainloader = torch.utils.data.DataLoader(trainset, batch_sampler=batch_sampler, **kwargs)
    testloader = torch.utils.data.DataLoader(totalset, batch_size=args.batchsize, shuffle=False, **kwargs)

    # setting up optimizer - the bias params should have twice the learning rate w.r.t. weights params
    bias_params = filter(lambda x: ('bias' in x[0]), net.named_parameters())
    bias_params = list(map(lambda x: x[1], bias_params))
    nonbias_params = filter(lambda x: ('bias' not in x[0]), net.named_parameters())
    nonbias_params = list(map(lambda x: x[1], nonbias_params))

    optimizer = optim.Adam([{'params': bias_params, 'lr': 2*args.lr},
                            {'params': nonbias_params},
                            {'params': criterion2.parameters(), 'lr': args.lr},
                            ], lr=args.lr, betas=(0.99, 0.999))

    # this is needed for WARM START
    if args.resume:
        filename = outputdir+'/FTcheckpoint_%d.pth.tar' % args.level
        if os.path.isfile(filename):
            print("==> loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)
            net.load_state_dict(checkpoint['state_dict'])
            criterion2.load_state_dict(checkpoint['criterion_state_dict'])
            startepoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
            _sigma1 = checkpoint['sigma1']
            _sigma2 = checkpoint['sigma2']
            _lambda = checkpoint['lambda']
            _delta = checkpoint['delta']
            _delta1 = checkpoint['delta1']
            _delta2 = checkpoint['delta2']
        else:
            print("==> no checkpoint found at '{}'".format(filename))
            raise

    # This is the actual Algorithm
    flag = 0
    for epoch in range(startepoch, args.nepoch):
        if args.tensorboard:
            log_value('sigma1', _sigma1, epoch)
            log_value('sigma2', _sigma2, epoch)
            log_value('lambda', _lambda, epoch)

        train(trainloader, net, optimizer, criterion1, criterion2, epoch, use_cuda, _sigma1, _sigma2, _lambda)
        Z, U, change_in_assign = test(testloader, net, criterion2, epoch, use_cuda, _delta, pairs, numeval, flag)

        if flag:
            # As long as the change in label assignment < threshold, DCC continues to run.
            # Note: This condition is always met in the very first epoch after the flag is set.
            # This false criterion is overwritten by checking for the condition twice.
            if change_in_assign > stopping_threshold:
                flag += 1
            if flag == 4:
                break

        if((epoch+1) % args.M == 0):
            _sigma1 = max(_delta1, _sigma1 / 2)
            _sigma2 = max(_delta2, _sigma2 / 2)
            if _sigma2 == _delta2 and flag == 0:
                # Start checking for stopping criterion
                flag = 1

        # Save checkpoint
        index = (epoch // args.M) * args.M
        save_checkpoint({'epoch': epoch+1,
                         'state_dict': net.state_dict(),
                         'criterion_state_dict': criterion2.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         'sigma1': _sigma1,
                         'sigma2': _sigma2,
                         'lambda': _lambda,
                         'delta': _delta,
                         'delta1': _delta1,
                         'delta2': _delta2,
                         }, index, filename=outputdir)

    sio.savemat(os.path.join(outputdir, 'features'), {'Z': Z, 'U': U, 'gtlabels': labels, 'w': pairs})


# Training
def train(trainloader, net, optimizer, criterion1, criterion2, epoch, use_cuda, _sigma1, _sigma2, _lambda):
    losses = AverageMeter()
    losses1 = AverageMeter()
    losses2 = AverageMeter()

    print('\n Epoch: %d' % epoch)

    net.train()

    for batch_idx, (inputs, pairweights, sampweights, pairs, index) in enumerate(trainloader):
        inputs = torch.squeeze(inputs,0)
        pairweights = torch.squeeze(pairweights)
        sampweights = torch.squeeze(sampweights)
        index = torch.squeeze(index)
        pairs = pairs.view(-1, 2)

        if use_cuda:
            inputs = inputs.cuda()
            pairweights = pairweights.cuda()
            sampweights = sampweights.cuda()
            index = index.cuda()
            pairs = pairs.cuda()

        optimizer.zero_grad()
        inputs_Var, sampweights, pairweights = Variable(inputs), Variable(sampweights, requires_grad=False), \
                                               Variable(pairweights, requires_grad=False)

        enc, dec = net(inputs_Var)
        loss1 = criterion1(inputs_Var, dec, sampweights)
        loss2 = criterion2(enc, sampweights, pairweights, pairs, index, _sigma1, _sigma2, _lambda)
        loss = loss1 + loss2

        # record loss
        losses1.update(loss1.data[0], inputs.size(0))
        losses2.update(loss2.data[0], inputs.size(0))
        losses.update(loss.data[0], inputs.size(0))

        loss.backward()
        optimizer.step()

    # log to TensorBoard
    if args.tensorboard:
        log_value('total_loss', losses.avg, epoch)
        log_value('reconstruction_loss', losses1.avg, epoch)
        log_value('dcc_loss', losses2.avg, epoch)


# Testing
def test(testloader, net, criterion, epoch, use_cuda, _delta, pairs, numeval, flag):
    net.eval()

    original = []
    features = []
    labels = []

    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs = inputs.cuda()
        inputs_Var = Variable(inputs, volatile=True)
        enc, dec = net(inputs_Var)
        features += list(enc.data.cpu().numpy())
        labels += list(targets)
        original += list(inputs.cpu().numpy())

    original, features, labels = np.asarray(original).astype(np.float32), np.asarray(features).astype(np.float32), \
                                  np.asarray(labels)

    U = criterion.U.data.cpu().numpy()

    change_in_assign = 0
    # logs clustering measures only if sigma2 has reached the minimum (delta2)
    if flag:
        index, ari, ami, nmi, acc, n_components = computeObj(U, pairs, _delta, labels, numeval)

        # log to TensorBoard
        change_in_assign = np.abs(oldassignment - index).sum()
        if args.tensorboard:
            log_value('ARI', ari, epoch)
            log_value('AMI', ami, epoch)
            log_value('NMI', nmi, epoch)
            log_value('ACC', acc, epoch)
            log_value('Numcomponents', n_components, epoch)
            log_value('labeldiff', change_in_assign, epoch)

        oldassignment[...] = index

    return features, U, change_in_assign

# Saving checkpoint
def save_checkpoint(state, index, filename):
    newfilename = os.path.join(filename, 'FTcheckpoint_%d.pth.tar' % index)
    torch.save(state, newfilename)

if __name__ == '__main__':
    main()