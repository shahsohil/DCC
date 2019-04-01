"""Example for doing all steps in code only (other examples require calling different files separately)"""
from config import cfg, get_data_dir
from easydict import EasyDict as edict
from edgeConstruction import compressed_data
import matplotlib.pyplot as plt
import data_params as dp
import make_data
import pretraining
import extract_feature
import copyGraph
import DCC

datadir = get_data_dir(dp.easy.name)
N = 600

# first create the data
X, labels = make_data.make_easy_visual_data(datadir, N)

# visualize data
# we know there are 3 classes
for c in range(3):
    x = X[labels==c,:]
    plt.scatter(x[:,0], x[:,1], label=str(c))
plt.legend()
plt.show()

# then construct mkNN graph
k = 50
compressed_data(dp.easy.name, N, k, preprocess='none', algo='kNN', isPCA=None, format='mat')

# then pretrain to get features
args = edict()
args.db = dp.easy.name
args.niter = 500
args.step = 300
args.lr = 0.001

# if we need to resume for faster debugging/results
args.resume = False
args.level = None

args.batchsize = 300
args.ngpu = 1
args.deviceID = 0
args.tensorboard = True
args.h5 = False
args.id = 6
args.dim = 2
args.manualSeed = cfg.RNG_SEED
args.clean_log = True

# if we comment out the next pretraining step, use the latest checkpoint
index = len(dp.easy.dim)-1
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

# actually do DCC
args.batchsize = cfg.PAIRS_PER_BATCH
args.nepoch = 500
args.M = 20
args.lr = 0.001
out = DCC.main(args)
