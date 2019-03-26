import os
import os.path as osp
from config import cfg, get_data_dir

import random
import argparse
import numpy as np
import scipy.io as sio
import h5py

from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.datasets.samples_generator import make_blobs


def make_reuters_data(path, N):
    did_to_cat = {}
    cat_list = ['CCAT', 'GCAT', 'MCAT', 'ECAT']
    with open(osp.join(path, 'rcv1-v2.topics.qrels')) as fin:
        for line in fin.readlines():
            line = line.strip().split(' ')
            cat = line[0]
            did = int(line[1])
            if cat in cat_list:
                did_to_cat[did] = did_to_cat.get(did, []) + [cat]
        for did in did_to_cat.keys():
            if len(did_to_cat[did]) > 1:
                del did_to_cat[did]

    dat_list = ['lyrl2004_tokens_test_pt0.dat',
                'lyrl2004_tokens_test_pt1.dat',
                'lyrl2004_tokens_test_pt2.dat',
                'lyrl2004_tokens_test_pt3.dat',
                'lyrl2004_tokens_train.dat']
    data = []
    target = []
    cat_to_cid = {'CCAT': 0, 'GCAT': 1, 'MCAT': 2, 'ECAT': 3}
    del did
    for dat in dat_list:
        with open(osp.join(path, dat)) as fin:
            for line in fin.readlines():
                if line.startswith('.I'):
                    if 'did' in locals():
                        assert doc != ''
                        if did_to_cat.has_key(did):
                            data.append(doc)
                            target.append(cat_to_cid[did_to_cat[did][0]])
                    did = int(line.strip().split(' ')[1])
                    doc = ''
                elif line.startswith('.W'):
                    assert doc == ''
                else:
                    doc += line

    assert len(data) == len(did_to_cat)

    X = CountVectorizer(dtype=np.float64, max_features=2000, max_df=0.90).fit_transform(data)
    Y = np.asarray(target)

    X = TfidfTransformer(norm='l2', sublinear_tf=True).fit_transform(X)
    X = np.asarray(X.todense())

    minmaxscale = MinMaxScaler().fit(X)
    X = minmaxscale.transform(X)

    p = np.random.permutation(X.shape[0])
    X = X[p]
    Y = Y[p]

    fo = h5py.File(osp.join(path, 'traindata.h5'), 'w')
    fo.create_dataset('X', data=X[:N * 6 / 7])
    fo.create_dataset('Y', data=Y[:N * 6 / 7])
    fo.close()

    fo = h5py.File(osp.join(path, 'testdata.h5'), 'w')
    fo.create_dataset('X', data=X[N * 6 / 7:N])
    fo.create_dataset('Y', data=Y[N * 6 / 7:N])
    fo.close()


def load_mnist(root, training):
    if training:
        data = 'train-images-idx3-ubyte'
        label = 'train-labels-idx1-ubyte'
        N = 60000
    else:
        data = 't10k-images-idx3-ubyte'
        label = 't10k-labels-idx1-ubyte'
        N = 10000
    with open(osp.join(root, data), 'rb') as fin:
        fin.seek(16, os.SEEK_SET)
        X = np.fromfile(fin, dtype=np.uint8).reshape((N, 28 * 28))
    with open(osp.join(root, label), 'rb') as fin:
        fin.seek(8, os.SEEK_SET)
        Y = np.fromfile(fin, dtype=np.uint8)
    return X, Y


def make_mnist_data(path, isconv=False):
    X, Y = load_mnist(path, True)
    X = X.astype(np.float64)
    X2, Y2 = load_mnist(path, False)
    X2 = X2.astype(np.float64)
    X3 = np.concatenate((X, X2), axis=0)

    minmaxscale = MinMaxScaler().fit(X3)

    X = minmaxscale.transform(X)
    if isconv:
        X = X.reshape((-1, 1, 28, 28))

    sio.savemat(osp.join(path, 'traindata.mat'), {'X': X, 'Y': Y})

    X2 = minmaxscale.transform(X2)
    if isconv:
        X2 = X2.reshape((-1, 1, 28, 28))

    sio.savemat(osp.join(path, 'testdata.mat'), {'X': X2, 'Y': Y2})


def make_misc_data(path, filename, dim, isconv=False):
    import cPickle
    fo = open(osp.join(path, filename), 'r')
    data = cPickle.load(fo)
    fo.close()
    X = data['data'].astype(np.float64)
    Y = data['labels']

    minmaxscale = MinMaxScaler().fit(X)
    X = minmaxscale.transform(X)

    p = np.random.permutation(X.shape[0])
    X = X[p]
    Y = Y[p]

    N = X.shape[0]

    if isconv:
        X = X.reshape((-1, dim[2], dim[0], dim[1]))
    save_misc_data(path, X, Y, N)


def make_easy_visual_data(path, N=600):
    """Make 3 clusters of 2D data where the cluster centers lie along a line.
    The latent variable would be just their x or y value since that uniquely defines their projection onto the line.
    """

    line = (1.5, 1)
    centers = [(m, m * line[0] + line[1]) for m in (-4, 0, 6)]
    cluster_std = [1, 1, 1.5]
    X, labels = make_blobs(n_samples=N, cluster_std=cluster_std, centers=centers, n_features=len(centers[0]))
    save_misc_data(path, X, labels, N)


def save_misc_data(path, X, Y, N):
    sio.savemat(osp.join(path, 'traindata.mat'), {'X': X[:N * 4 / 5], 'Y': Y[:N * 4 / 5]})
    sio.savemat(osp.join(path, 'testdata.mat'), {'X': X[N * 4 / 5:], 'Y': Y[N * 4 / 5:]})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', dest='db', type=str, default='mnist', help='name of the dataset')

    args = parser.parse_args()
    np.random.seed(cfg.RNG_SEED)
    random.seed(cfg.RNG_SEED)

    datadir = get_data_dir(args.db)
    strpath = osp.join(datadir, 'traindata.mat')

    if not os.path.exists(strpath):
        if args.db == 'mnist':
            make_mnist_data(datadir)
        elif args.db == 'reuters':
            make_reuters_data(datadir, 10000)
        elif args.db == 'ytf':
            make_misc_data(datadir, 'YTFrgb.pkl', [55, 55, 3])
        elif args.db == 'coil100':
            make_misc_data(datadir, 'coil100rgb.pkl', [128, 128, 3])
        elif args.db == 'yale':
            make_misc_data(datadir, 'yale_DoG.pkl', [168, 192, 1])
        elif args.db == 'rcv1':
            make_misc_data(datadir, 'reuters.pkl', [1, 1, 2000])
        elif args.db == 'cmnist':
            make_mnist_data(datadir, isconv=True)
        elif args.db == 'cytf':
            make_misc_data(datadir, 'YTFrgb.pkl', [55, 55, 3], isconv=True)
        elif args.db == 'ccoil100':
            make_misc_data(datadir, 'coil100rgb.pkl', [128, 128, 3], isconv=True)
        elif args.db == 'cyale':
            make_misc_data(datadir, 'yale_DoG.pkl', [168, 192, 1], isconv=True)
        elif args.db == 'easy':
            make_easy_visual_data(datadir)
