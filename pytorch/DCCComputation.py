import os
import numpy as np

import scipy.io as sio
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix, diags
import scipy.sparse as sparse
from scipy.sparse.csgraph import connected_components
from sklearn import metrics

from config import cfg, get_data_dir

def makeDCCinp(args):
    # pretrained.mat or pretrained.h5 must be placed under the ../data/"db"/ directory. "db" stands for dataset
    datadir = get_data_dir(args.db)
    datafile = 'pretrained'

    if args.h5:
        datafile = os.path.join(datadir, datafile+'.h5')
    else:
        datafile = os.path.join(datadir, datafile+'.mat')
    assert os.path.exists(datafile), 'Training data not found at `{:s}`'.format(datafile)

    if args.h5:
        import h5py
        raw_data = h5py.File(datafile, 'r')
    else:
        raw_data = sio.loadmat(datafile, mat_dtype=True)

    data = raw_data['X'][:].astype(np.float32)
    Z = raw_data['Z'][:].astype(np.float32)
    # correct special case where Z is N x 1 and it gets loaded as 1 x N
    if Z.shape[0] == 1:
        Z = np.transpose(Z)

    labels = np.squeeze(raw_data['gtlabels'][:])
    pairs = raw_data['w'][:, :2].astype(int)

    if args.h5:
        raw_data.close()

    print('\n Loaded `{:s}` dataset for finetuning'.format(args.db))

    numpairs = pairs.shape[0]
    numsamples = data.shape[0]

    # Creating pairwise weights and individual sample sample for reconstruction loss term
    R = csr_matrix((np.ones(numpairs, dtype=np.float32), (pairs[:, 0], pairs[:, 1])), shape=(numsamples, numsamples))
    R = R + R.transpose()
    nconn = np.squeeze(np.array(np.sum(R, 1)))
    weights = np.average(nconn) / np.sqrt(nconn[pairs[:, 0]] * nconn[pairs[:, 1]])
    pairs = np.hstack((pairs, np.atleast_2d(weights).transpose()))

    return data, labels, pairs, Z, nconn


def computeHyperParams(pairs, Z):
    numpairs = len(pairs)
    numsamples = len(Z)
    epsilon = np.linalg.norm(Z[pairs[:, 0].astype(int)] - Z[pairs[:, 1].astype(int)], axis=1)
    epsilon = np.sort(epsilon / np.sqrt(cfg.DIM))
    # largest is noise just consider as noise
    if epsilon[-1] < cfg.RCC.NOISE_THRESHOLD:
        epsilon = np.asarray([cfg.RCC.NOISE_THRESHOLD])
    else:
        epsilon = epsilon[np.where(epsilon > cfg.RCC.NOISE_THRESHOLD)]

    # threshold for finding connected components
    robsamp = int(numpairs * cfg.RCC.MIN_RATIO_SAMPLES_DELTA)
    _delta = np.average(epsilon[:robsamp])

    robsamp = min(cfg.RCC.MAX_NUM_SAMPLES_DELTA, robsamp)
    _delta2 = float(np.average(epsilon[:robsamp]) / 2)
    _sigma2 = float(3 * (epsilon[-1] ** 2))

    _delta1 = float(np.average(np.linalg.norm(Z - np.average(Z, axis=0)[np.newaxis, :], axis=1) ** 2))
    _sigma1 = float(max(cfg.RCC.GNC_DATA_START_POINT, 16 * _delta1))

    print('The endpoints are Delta1: {:.3f}, Delta2: {:.3f}'.format(_delta1, _delta2))

    lmdb = np.ones(numpairs, dtype=np.float32)
    lmdb_data = np.ones(numsamples, dtype=np.float32)
    _lambda = compute_lambda(pairs, Z, lmdb, lmdb_data)

    return _sigma1, _sigma2, _lambda, _delta, _delta1, _delta2, lmdb, lmdb_data


def compute_lambda(pairs, Z, lmdb, lmdb_data):
    numsamples = len(Z)

    R = csr_matrix((lmdb * pairs[:,2], (pairs[:,0].astype(int), pairs[:,1].astype(int))), shape=(numsamples, numsamples))
    R = R + R.transpose()

    D = diags(np.squeeze(np.array(np.sum(R,1))), 0)
    I = diags(lmdb_data, 0)

    spndata = np.linalg.norm(I * Z, ord=2)
    eiglmdbdata,_ = sparse.linalg.eigsh(I, k=1)
    eigM,_ = sparse.linalg.eigsh(D - R, k=1)

    _lambda = float(spndata / (eiglmdbdata + eigM))

    return _lambda


def computeObj(U, pairs, _delta, gtlabels, numeval):
    """ This is similar to computeObj function in Matlab """
    numsamples = len(U)
    diff = np.linalg.norm(U[pairs[:, 0].astype(int)] - U[pairs[:, 1].astype(int)], axis=1)**2

    # computing clustering measures
    index1 = np.sqrt(diff) < _delta
    index = np.where(index1)
    adjacency = csr_matrix((np.ones(len(index[0])), (pairs[index[0], 0].astype(int), pairs[index[0], 1].astype(int))),
                           shape=(numsamples, numsamples))
    adjacency = adjacency + adjacency.transpose()
    n_components, labels = connected_components(adjacency, directed=False)

    index2 = labels[pairs[:, 0].astype(int)] == labels[pairs[:, 1].astype(int)]

    ari, ami, nmi, acc = benchmarking(gtlabels[:numeval], labels[:numeval])

    return index2, ari, ami, nmi, acc, n_components, labels


def benchmarking(gtlabels, labels):
    # TODO: Please note that the AMI definition used in the paper differs from that in the sklearn python package.
    # TODO: Please modify it accordingly.
    numeval = len(gtlabels)
    ari = metrics.adjusted_rand_score(gtlabels[:numeval], labels[:numeval])
    ami = metrics.adjusted_mutual_info_score(gtlabels[:numeval], labels[:numeval])
    nmi = metrics.normalized_mutual_info_score(gtlabels[:numeval], labels[:numeval])
    acc = clustering_accuracy(gtlabels[:numeval], labels[:numeval])

    return ari, ami, nmi, acc


def clustering_accuracy(gtlabels, labels):
    cost_matrix = []
    categories = np.unique(gtlabels)
    nr = np.amax(labels) + 1
    for i in np.arange(len(categories)):
        cost_matrix.append(np.bincount(labels[gtlabels == categories[i]], minlength=nr))
    cost_matrix = np.asarray(cost_matrix).T
    row_ind, col_ind = linear_sum_assignment(np.max(cost_matrix) - cost_matrix)

    return float(cost_matrix[row_ind, col_ind].sum()) / len(gtlabels)