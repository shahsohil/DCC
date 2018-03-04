import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

# This class is similar to SDAE code.
# This model is initiated when SDAE is needed for training without Dropout modules i.e., during DCC.
class extractSDAE(nn.Module):
    def __init__(self, dim, slope=0.0):
        super(extractSDAE, self).__init__()
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

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.normal(m.weight, std=1e-2)
                if m.bias.data is not None:
                    init.constant(m.bias, 0)

    def forward(self,x):
        inp = x.view(-1, self.in_dim)
        encoded = inp
        for i, encoder in enumerate(self.enc):
            encoded = encoder(encoded)
            if i < self.nlayers-1:
                encoded = F.leaky_relu(encoded, negative_slope=self.reluslope)
        out = encoded
        for i, decoder in reversed(list(enumerate(self.dec))):
            out = decoder(out)
            if i:
                out = F.leaky_relu(out, negative_slope=self.reluslope)
        return encoded, out

def extract_sdae_mnist(slope=0.0, dim=10):
    return extractSDAE(dim=[784, 500, 500, 2000, dim], slope=slope)

def extract_sdae_reuters(slope=0.0, dim=10):
    return extractSDAE(dim=[2000, 500, 500, 2000, dim], slope=slope)

def extract_sdae_ytf(slope=0.0, dim=10):
    return extractSDAE(dim=[9075, 500, 500, 2000, dim], slope=slope)

def extract_sdae_coil100(slope=0.0, dim=10):
    return extractSDAE(dim=[49152, 500, 500, 2000, dim], slope=slope)

def extract_sdae_yale(slope=0.0, dim=10):
    return extractSDAE(dim=[32256, 500, 500, 2000, dim], slope=slope)