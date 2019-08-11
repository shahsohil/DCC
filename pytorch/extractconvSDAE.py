import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class extractconvSDAE(nn.Module):
    def __init__(self, dim, output_padding, numpen, slope=0.0):
        super(extractconvSDAE, self).__init__()
        self.in_dim = dim[0]
        self.nlayers = len(dim)-1
        self.reluslope = slope
        self.numpen = numpen
        self.enc, self.dec = [], []
        self.benc, self.bdec = [], []
        for i in range(self.nlayers):
            if i == self.nlayers - 1:
                self.enc.append(nn.Linear(dim[i]*numpen*numpen, dim[i+1]))
                self.benc.append(nn.BatchNorm2d(dim[i + 1]))
                self.dec.append(nn.ConvTranspose2d(dim[i + 1], dim[i], kernel_size=numpen, stride=1))
                self.bdec.append(nn.BatchNorm2d(dim[i]))
            elif i == 0:
                self.enc.append(nn.Conv2d(dim[i], dim[i + 1], kernel_size=4, stride=2, padding=1))
                self.benc.append(nn.BatchNorm2d(dim[i + 1]))
                self.dec.append(nn.ConvTranspose2d(dim[i+1], dim[i], kernel_size=4, stride=2, padding=1,
                                                   output_padding=output_padding[i]))
                self.bdec.append(nn.BatchNorm2d(dim[i]))
            else:
                self.enc.append(nn.Conv2d(dim[i], dim[i + 1], kernel_size=5, stride=2, padding=2))
                self.benc.append(nn.BatchNorm2d(dim[i + 1]))
                self.dec.append(nn.ConvTranspose2d(dim[i+1], dim[i], kernel_size=5, stride=2, padding=2,
                                                   output_padding=output_padding[i]))
                self.bdec.append(nn.BatchNorm2d(dim[i]))
            setattr(self, 'enc_{}'.format(i), self.enc[-1])
            setattr(self, 'benc_{}'.format(i), self.benc[-1])
            setattr(self, 'dec_{}'.format(i), self.dec[-1])
            setattr(self, 'bdec_{}'.format(i), self.bdec[-1])
        self.base = []
        self.bbase = []
        for i in range(self.nlayers):
            self.base.append(nn.Sequential(*self.enc[:i]))
            self.bbase.append(nn.Sequential(*self.benc[:i]))

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.normal(m.weight, std=1e-2)
                if m.bias.data is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias.data is not None:
                    init.constant(m.bias, 0)

    def forward(self,x):
        encoded = x
        for i, (encoder,bencoder) in enumerate(zip(self.enc,self.benc)):
            if i == self.nlayers-1:
                encoded = encoded.view(encoded.size(0), -1)
            encoded = encoder(encoded)
            if i < self.nlayers-1:
                encoded = bencoder(encoded)
                encoded = F.leaky_relu(encoded, negative_slope=self.reluslope)
        out = encoded
        for i, (decoder,bdecoder) in reversed(list(enumerate(zip(self.dec,self.bdec)))):
            if i == self.nlayers-1:
                out = out.view(out.size(0), -1, 1, 1)
            out = decoder(out)
            if i:
                out = bdecoder(out)
                out = F.leaky_relu(out, negative_slope=self.reluslope)
        return encoded, out

