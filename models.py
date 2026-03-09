import math
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Sampler(nn.Module):
    def __init__(self):
        super(Sampler, self).__init__()

    def forward(self, input_x):
        mu = input_x[0]
        logvar = input_x[1]

        std = logvar.mul(0.5).exp_()
        eps = torch.randn(std.size(), device=std.device)
        eps = Variable(eps)
        return eps.mul(std).add_(mu)


class Encoder(nn.Module):
    def __init__(self, image_size, nz, ngf, nc):
        super(Encoder, self).__init__()

        n = math.log2(image_size)

        assert n == round(n), 'imageSize must be a power of 2'
        assert n >= 3, 'imageSize must be at least 8'
        n = int(n)

        self.conv1 = nn.Conv2d(ngf * 2 ** (n - 3), nz, 4)
        self.conv2 = nn.Conv2d(ngf * 2 ** (n - 3), nz, 4)

        # build a conventional convolutional encoder (same depth as before)
        self.encoder = nn.Sequential()
        # first block: input_x -> ngf
        self.encoder.add_module('conv1', nn.Conv2d(nc, ngf, 4, 2, 1, bias=False))
        self.encoder.add_module('lrelu1', nn.LeakyReLU(0.2, inplace=True))
        # subsequent blocks doubling channels each time
        for i in range(n - 3):
            in_ch = ngf * 2 ** i
            out_ch = ngf * 2 ** (i + 1)
            idx = i + 2
            self.encoder.add_module('conv%d' % idx, nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False))
            self.encoder.add_module('bn%d' % idx, nn.BatchNorm2d(out_ch))
            self.encoder.add_module('lrelu%d' % idx, nn.LeakyReLU(0.2, inplace=True))

    def forward(self, input_x):
        output = self.encoder(input_x)
        return [self.conv1(output), self.conv2(output)]


class Generator(nn.Module):
    def __init__(self, nz, ngf, nc, ngpu=1):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        # DCGAN generator blocks (matches dcgan.py)
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Tanh()
        )

    def forward(self, input_x):
        if isinstance(input_x.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input_x, range(self.ngpu))
        else:
            output = self.main(input_x)
        return output


class Discriminator(nn.Module):
    def __init__(self, image_size, ndf, nc, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        n = math.log2(image_size)

        assert n == round(n), 'imageSize must be a power of 2'
        assert n >= 3, 'imageSize must be at least 8'
        n = int(n)
        # build discriminator dynamically so it supports multiple image sizes
        self.main = nn.Sequential()
        # input_x conv
        self.main.add_module('input-conv', nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        self.main.add_module('input-lrelu', nn.LeakyReLU(0.2, inplace=True))
        # add pyramid blocks based on image size
        for i in range(n - 3):
            in_ch = ndf * 2 ** i
            out_ch = ndf * 2 ** (i + 1)
            self.main.add_module('conv%d' % i, nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False))
            self.main.add_module('bn%d' % i, nn.BatchNorm2d(out_ch))
            self.main.add_module('lrelu%d' % i, nn.LeakyReLU(0.2, inplace=True))

        # final output conv to single score (kernel should match remaining spatial size)
        self.main.add_module('output-conv', nn.Conv2d(ndf * 2 ** (n - 3), 1, 4, 1, 0, bias=False))
        self.main.add_module('output-sigmoid', nn.Sigmoid())

    def forward(self, input_x):
        if isinstance(input_x.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input_x, range(self.ngpu))
        else:
            output = self.main(input_x)

        return output.view(-1, 1)
