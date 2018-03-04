import os.path as osp
from easydict import EasyDict as edict

__C = edict()

cfg = __C

#######
# OPTIONS FROM RCC CODE
#######
__C.RCC = edict()

__C.RCC.NOISE_THRESHOLD = 0.01

__C.RCC.MAX_NUM_SAMPLES_DELTA = 250

__C.RCC.MIN_RATIO_SAMPLES_DELTA = 0.01

__C.RCC.GNC_DATA_START_POINT = 132*16

#######
# MISC OPTIONS
#######

# For reproducibility
__C.RNG_SEED = 50

# Root directory
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__),'..'))

# size of the dataset
__C.SAMPLES = 70000

# embedding dimension
__C.DIM = 10

# Number of pairs per batch
__C.PAIRS_PER_BATCH = 128

# Fraction of "change in label assignment of pairs" to be considered for stopping criterion - 1% of pairs
__C.STOPPING_CRITERION = 0.001


def get_data_dir(db):
    """
    :param db:
    :return: path to data directory
    """
    path = osp.abspath(osp.join(__C.ROOT_DIR, 'data', db))
    return path

def get_output_dir(db):
    """
    :param db:
    :return: path to data directory
    """
    path = osp.abspath(osp.join(__C.ROOT_DIR, 'data', db, 'results'))
    return path

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count