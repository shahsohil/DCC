from __future__ import print_function
import os
import random
import numpy as np
import argparse
from config import cfg, get_data_dir, get_output_dir, AverageMeter
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import data_params as dp

from SDAE import sdae_mnist, sdae_reuters, sdae_ytf, sdae_coil100, sdae_yale, sdae_easy
from convSDAE import convsdae_mnist, convsdae_coil100, convsdae_ytf, convsdae_yale
from custom_data import DCCPT_data

# used for logging to TensorBoard
from tensorboard_logger import Logger

# Parse all the input argument
parser = argparse.ArgumentParser(description='PyTorch SDAE Training')
parser.add_argument('--batchsize', type=int, default=256, help='batch size used for pretraining')
# mnist=50000, ytf=6700, coil100=5000, reuters10k=50000, yale=10000, rcv1=6100. This amounts to ~200 epochs.
parser.add_argument('--niter', type=int, default=50000, help='number of iterations used for pretraining')
# mnist=20000, ytf=2700, coil100=2000, reuters10k=20000, yale=4000, rcv1=2450. This amounts to ~80 epochs.
parser.add_argument('--step', type=int, default=20000,
                    help='stepsize in terms of number of iterations for pretraining. lr is decreased by 10 after every stepsize.')
# Note: The learning rate of pretraining stage differs for each dataset.
# As noted in the paper, it depends on the original dimension of the data samples.
# This is purely selected such that the SDAE's are trained with maximum possible learning rate for each dataset.
# We set mnist,reuters,rcv1=10, ytf=1, coil100,yaleb=0.1
# For convolutional SDAE lr if fixed to 0.1
parser.add_argument('--lr', default=10, type=float, help='initial learning rate for pretraining')
parser.add_argument('--manualSeed', default=cfg.RNG_SEED, type=int, help='manual seed')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--level', default=0, type=int, help='index of the module to resume from')
parser.add_argument('--data', dest='db', type=str, default='mnist', help='name of the dataset')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--dim', type=int, help='dimension of embedding space', default=10)
parser.add_argument('--deviceID', type=int, help='deviceID', default=0)
parser.add_argument('--h5', dest='h5', help='to store as h5py file', default=False, type=bool)
parser.add_argument('--tensorboard', help='Log progress to TensorBoard', action='store_true')
parser.add_argument('--id', type=int, help='identifying number for storing tensorboard logs')

def main(args):
    datadir = get_data_dir(args.db)
    outputdir = get_output_dir(args.db)

    logger = None
    if args.tensorboard:
        # One should create folder for storing logs
        loggin_dir = os.path.join(outputdir, 'runs', 'pretraining')
        if not os.path.exists(loggin_dir):
            os.makedirs(loggin_dir)
        loggin_dir = os.path.join(loggin_dir, '%s' % (args.id))
        logger = Logger(loggin_dir)

    use_cuda = torch.cuda.is_available()

    # Set the seed for reproducing the results
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.manualSeed)
        torch.backends.cudnn.enabled = True
        cudnn.benchmark = True

    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}
    trainset = DCCPT_data(root=datadir, train=True, h5=args.h5)
    testset = DCCPT_data(root=datadir, train=False, h5=args.h5)

    nepoch = int(np.ceil(np.array(args.niter * args.batchsize, dtype=float) / len(trainset)))
    step = int(np.ceil(np.array(args.step * args.batchsize, dtype=float) / len(trainset)))

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize, shuffle=True, **kwargs)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=True, **kwargs)

    return pretrain(args, outputdir, {'nlayers':4, 'dropout':0.2, 'reluslope':0.0,
                       'nepoch':nepoch, 'lrate':[args.lr], 'wdecay':[0.0], 'step':step}, use_cuda, trainloader, testloader, logger)

def pretrain(args, outputdir, params, use_cuda, trainloader, testloader, logger):
    numlayers = params['nlayers']
    lr = params['lrate'][0]
    maxepoch = params['nepoch']
    stepsize = params['step']
    startlayer = 0

    # For simplicity, I have created placeholder for each datasets and model
    if args.db == 'mnist':
        net = sdae_mnist(dropout=params['dropout'], slope=params['reluslope'], dim=args.dim)
    elif args.db == 'reuters' or args.db == 'rcv1':
        net = sdae_reuters(dropout=params['dropout'], slope=params['reluslope'], dim=args.dim)
    elif args.db == 'ytf':
        net = sdae_ytf(dropout=params['dropout'], slope=params['reluslope'], dim=args.dim)
    elif args.db == 'coil100':
        net = sdae_coil100(dropout=params['dropout'], slope=params['reluslope'], dim=args.dim)
    elif args.db == 'yale':
        net = sdae_yale(dropout=params['dropout'], slope=params['reluslope'], dim=args.dim)
    elif args.db == 'cmnist':
        net = convsdae_mnist(dropout=params['dropout'], slope=params['reluslope'])
    elif args.db == 'ccoil100':
        net = convsdae_coil100(dropout=params['dropout'], slope=params['reluslope'])
        numlayers = 6
    elif args.db == 'cytf':
        net = convsdae_ytf(dropout=params['dropout'], slope=params['reluslope'])
        numlayers = 5
    elif args.db == 'cyale':
        net = convsdae_yale(dropout=params['dropout'], slope=params['reluslope'])
        numlayers = 6
    elif args.db == 'easy':
        net = sdae_easy(dropout=params['dropout'], slope=params['reluslope'], dim=args.dim)
        numlayers = len(dp.easy.dim)
    else:
        raise ValueError("Unexpected database %s" % args.db)

    # For the final FT stage of SDAE pretraining, the total epoch is twice that of previous stages.
    maxepoch = [maxepoch]*numlayers + [maxepoch*2]
    stepsize = [stepsize]*(numlayers+1)

    if args.resume:
        filename = outputdir+'/checkpoint_%d.pth.tar' % args.level
        if os.path.isfile(filename):
            print("==> loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)
            net.load_state_dict(checkpoint['state_dict'])
            startlayer = args.level+1
        else:
            print("==> no checkpoint found at '{}'".format(filename))
            raise

    if use_cuda:
        net.cuda()

    for index in range(startlayer, numlayers+1):
        # Freezing previous layer weights
        if index < numlayers:
            for par in net.base[index].parameters():
                par.requires_grad = False
            if args.db == 'cmnist' or args.db == 'ccoil100' or args.db == 'cytf' or args.db == 'cyale':
                for par in net.bbase[index].parameters():
                    par.requires_grad = False
                for m in net.bbase[index].modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.training = False
        else:
            for par in net.base[numlayers-1].parameters():
                par.requires_grad = True
            if args.db == 'cmnist' or args.db == 'ccoil100' or args.db == 'cytf' or args.db == 'cyale':
                for par in net.bbase[numlayers-1].parameters():
                    par.requires_grad = True
                for m in net.bbase[numlayers-1].modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.training = True

        # setting up optimizer - the bias params should have twice the learning rate w.r.t. weights params
        bias_params = filter(lambda x: ('bias' in x[0]) and (x[1].requires_grad), net.named_parameters())
        bias_params = list(map(lambda x: x[1], bias_params))
        nonbias_params = filter(lambda x: ('bias' not in x[0]) and (x[1].requires_grad), net.named_parameters())
        nonbias_params = list(map(lambda x: x[1], nonbias_params))

        optimizer = optim.SGD([{'params': bias_params, 'lr': 2*lr}, {'params': nonbias_params}],
                              lr=lr, momentum=0.9, weight_decay=params['wdecay'][0], nesterov=True)

        scheduler = lr_scheduler.StepLR(optimizer, step_size=stepsize[index], gamma=0.1)

        print('\nIndex: %d \t Maxepoch: %d'%(index, maxepoch[index]))

        for epoch in range(maxepoch[index]):
            scheduler.step()
            train(trainloader, net, index, optimizer, epoch, use_cuda, logger)
            test(testloader, net, index, epoch, use_cuda, logger)
            # Save checkpoint
            save_checkpoint({'epoch': epoch+1, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict()},
                            index, filename=outputdir)
    return index, net


# Training
def train(trainloader, net, index, optimizer, epoch, use_cuda, logger):
    losses = AverageMeter()

    print('\nIndex: %d \t Epoch: %d' %(index,epoch))

    net.train()

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs = inputs.cuda()
        optimizer.zero_grad()
        inputs_Var = Variable(inputs)
        outputs = net(inputs_Var, index)

        # record loss
        losses.update(outputs.item(), inputs.size(0))

        outputs.backward()
        optimizer.step()

    # log to TensorBoard
    if logger:
        logger.log_value('train_loss_{}'.format(index), losses.avg, epoch)


# Testing
def test(testloader, net, index, epoch, use_cuda, logger):
    losses = AverageMeter()

    net.eval()

    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs = inputs.cuda()
        inputs_Var = Variable(inputs, volatile=True)
        outputs = net(inputs_Var, index)

        # measure accuracy and record loss
        losses.update(outputs.item(), inputs.size(0))

    # log to TensorBoard
    if logger:
        logger.log_value('val_loss_{}'.format(index), losses.avg, epoch)


# Saving checkpoint
def save_checkpoint(state, index, filename):
    torch.save(state, filename+'/checkpoint_%d.pth.tar' % index)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)