import torch
import torch.nn as nn
import numpy as np

class DCCWeightedELoss(nn.Module):
    def __init__(self, size_average=True):
        super(DCCWeightedELoss, self).__init__()
        self.size_average = size_average

    def forward(self, inputs, outputs, weights):
        out = (inputs - outputs).view(len(inputs), -1)
        out = torch.sum(weights * torch.norm(out, p=2, dim=1)**2)

        assert np.isfinite(out.data.cpu().numpy()).all(), 'Nan found in data'

        if self.size_average:
            out = out / inputs.nelement()

        return out

class DCCLoss(nn.Module):
    def __init__(self, nsamples, ndim, initU, size_average=True):
        super(DCCLoss, self).__init__()
        self.dim = ndim
        self.nsamples = nsamples
        self.size_average = size_average
        self.U = nn.Parameter(torch.Tensor(self.nsamples, self.dim))
        self.reset_parameters(initU+1e-6*np.random.randn(*initU.shape).astype(np.float32))

    def reset_parameters(self, initU):
        assert np.isfinite(initU).all(), 'Nan found in initialization'
        self.U.data = torch.from_numpy(initU)

    def forward(self, enc_out, sampweights, pairweights, pairs, index, _sigma1, _sigma2, _lambda):
        centroids = self.U[index]

        # note that sigmas here are labelled mu in the paper
        # data loss
        # enc_out is Y, the original embedding without shift
        out1 = torch.norm((enc_out - centroids).view(len(enc_out), -1), p=2, dim=1) ** 2
        out11 = torch.sum(_sigma1 * sampweights * out1 / (_sigma1 + out1))

        # pairwise loss
        out2 = torch.norm((centroids[pairs[:, 0]] - centroids[pairs[:, 1]]).view(len(pairs), -1), p=2, dim=1) ** 2

        out21 = _lambda * torch.sum(_sigma2 * pairweights * out2 / (_sigma2 + out2))

        out = out11 + out21

        if self.size_average:
            out = out / enc_out.nelement()

        return out