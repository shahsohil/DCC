"""Example for doing all steps in code only (other examples require calling different files separately)"""
from config import cfg, get_data_dir
from easydict import EasyDict as edict
from edgeConstruction import compressed_data
import data_params as dp
import make_data
import pretraining
import extract_feature
import copyGraph

datadir = get_data_dir(dp.easy.name)
N = 600

# first create the data
make_data.make_easy_visual_data(datadir, N)

# then construct mkNN graph
compressed_data(dp.easy.name, N, 10, preprocess='none', algo='mkNN', isPCA=None, format='mat')

# then pretrain to get features
args = edict()
args.db = dp.easy.name
args.niter = 500
args.step = 300
args.lr = 0.001
args.resume = False
args.batchsize = 300
args.ngpu = 1
args.deviceID = 0
args.tensorboard = True
args.h5 = False
args.id = 4
args.dim = 2
args.manualSeed = cfg.RNG_SEED
index, _ = pretraining.main(args)

# extract pretrained features
args.feat = 'pretrained'
args.torchmodel = 'checkpoint_{}.pth.tar'.format(index)
extract_feature.main(args)

# merge the features and mkNN graph
args.g = 'pretrained.mat'
args.out = 'pretrained'
args.feat = 'pretrained.pkl'
copyGraph.main(args)