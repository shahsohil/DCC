import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

# The model definition for Stacked Denoising AE.
# This model is used during the pretraining stage.
class SDAE(nn.Module):
    def __init__(self, dim, dropout=0.2, slope=0.0):
        super(SDAE, self).__init__()
        self.in_dim = dim[0]
        self.nlayers = len(dim)-1
        self.reluslope = slope
        self.enc, self.dec = [], []
        for i in range(self.nlayers):
            self.enc.append(nn.Linear(dim[i], dim[i+1]))
            setattr(self, 'enc_{}'.format(i), self.enc[-1])
            self.dec.append(nn.Linear(dim[i+1], dim[i]))
            setattr(self, 'dec_{}'.format(i), self.dec[-1])
        self.base = []
        for i in range(self.nlayers):
            self.base.append(nn.Sequential(*self.enc[:i]))
        self.dropmodule1 = nn.Dropout(p=dropout)
        self.dropmodule2 = nn.Dropout(p=dropout)
        self.loss = nn.MSELoss(size_average=True)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.normal(m.weight, std=1e-2)
                if m.bias.data is not None:
                    init.constant(m.bias, 0)

    def forward(self,x,index):
        inp = x.view(-1, self.in_dim)
        encoded = inp
        for i, encoder in enumerate(self.enc):
            if i < index:
                encoded = encoder(encoded)
                if i < self.nlayers-1:
                    encoded = F.leaky_relu(encoded, negative_slope=self.reluslope)
            if i == index:
                inp = encoded
                out = encoded
                if index:
                    out = self.dropmodule1(out)
                out = encoder(out)
        if index < self.nlayers-1:
            out = F.leaky_relu(out, negative_slope=self.reluslope)
            out = self.dropmodule2(out)
        if index >= self.nlayers:
            out = encoded
        for i, decoder in reversed(list(enumerate(self.dec))):
            if index >= self.nlayers:
                out = decoder(out)
                if i:
                    out = F.leaky_relu(out, negative_slope=self.reluslope)
            if i == index:
                out = decoder(out)
                if index:
                    out = F.leaky_relu(out, negative_slope=self.reluslope)
        out = self.loss(out, inp)
        return out

def sdae_mnist(dropout=0.2, slope=0.0, dim=10):
    return SDAE(dim=[784, 500, 500, 2000, dim], dropout=dropout, slope=slope)

def sdae_reuters(dropout=0.2, slope=0.0, dim=10):
    return SDAE(dim=[2000, 500, 500, 2000, dim], dropout=dropout, slope=slope)

def sdae_ytf(dropout=0.2, slope=0.0, dim=10):
    return SDAE(dim=[9075, 500, 500, 2000, dim], dropout=dropout, slope=slope)

def sdae_coil100(dropout=0.2, slope=0.0, dim=10):
    return SDAE(dim=[49152, 500, 500, 2000, dim], dropout=dropout, slope=slope)

def sdae_yale(dropout=0.2, slope=0.0, dim=10):
    return SDAE(dim=[32256, 500, 500, 2000, dim], dropout=dropout, slope=slope)

def sdae_easy(dropout=0.2, slope=0.0, dim=1):
    return SDAE(dim=[2, 4, dim], dropout=dropout, slope=slope)