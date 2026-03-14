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

        std = logvar.mul(0.5).exp()
        eps = torch.randn(std.size(), device=std.device)
        eps = Variable(eps)
        return eps.mul(std).add(mu)


class Encoder(nn.Module):
    def __init__(self, image_size, nz, ngf, nc):
        super(Encoder, self).__init__()

        n = math.log2(image_size)
        assert n == round(n), 'imageSize must be a power of 2'
        assert n >= 3, 'imageSize must be at least 8'
        n = int(n)

        # Architecture per VAE-GAN figure (5x5 convs, stride 2, BNorm, ReLU)
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 64, 5, 2, 2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 128, 5, 2, 2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
            nn.Conv2d(128, 256, 5, 2, 2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
        )

        spatial = image_size // (2 ** 3)
        self.fc = nn.Sequential(
            nn.Linear(256 * spatial * spatial, 2048, bias=False),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=False),
        )
        self.fc_mu = nn.Linear(2048, nz)
        self.fc_logvar = nn.Linear(2048, nz)

    def forward(self, input_x):
        output = self.encoder(input_x)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return [self.fc_mu(output), self.fc_logvar(output)]


class Generator(nn.Module):
    def __init__(self, nz, ngf, nc, ngpu=1, image_size=32):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        n = math.log2(image_size)
        assert n == round(n), 'imageSize must be a power of 2'
        assert n >= 3, 'imageSize must be at least 8'
        n = int(n)

        # Architecture per VAE-GAN figure (8x8x256 FC, 5x5 deconvs, tanh)
        self.start_spatial = image_size // (2 ** 3)
        self.fc = nn.Sequential(
            nn.Linear(nz, 256 * self.start_spatial * self.start_spatial, bias=False),
            nn.BatchNorm1d(256 * self.start_spatial * self.start_spatial),
            nn.ReLU(inplace=False),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 5, 2, 2, output_padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(256, 128, 5, 2, 2, output_padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(128, 32, 5, 2, 2, output_padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.Conv2d(32, nc, 5, 1, 2, bias=False),
            nn.Tanh(),
        )

    def forward(self, input_x):
        if input_x.dim() == 4 and input_x.size(2) == 1 and input_x.size(3) == 1:
            input_x = input_x.view(input_x.size(0), -1)

        if isinstance(input_x.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.fc, input_x, range(self.ngpu))
        else:
            output = self.fc(input_x)
        output = output.view(output.size(0), 256, self.start_spatial, self.start_spatial)
        if isinstance(output.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.deconv, output, range(self.ngpu))
        else:
            output = self.deconv(output)
        return output


class Discriminator(nn.Module):
    def __init__(self, image_size, ndf, nc, ngpu, hook_layers=None):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        n = math.log2(image_size)

        assert n == round(n), 'imageSize must be a power of 2'
        assert n >= 3, 'imageSize must be at least 8'
        n = int(n)
        # Architecture per VAE-GAN figure (5x5 convs, stride 2, BNorm, ReLU)
        self.main = nn.Sequential(
            nn.Conv2d(nc, 32, kernel_size = 5, stride = 1, padding = 2, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv2d(32, 128, kernel_size = 5, stride = 2, padding = 2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
            nn.Conv2d(128, 256, kernel_size = 5, stride = 2, padding = 2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size = 5, stride = 2, padding = 2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
        )

        spatial = image_size // (2 ** 3)
        self.classifier = nn.Sequential(
            nn.Linear(256 * spatial * spatial, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=False),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

        self._hook_handles = []
        self._hooked = {}
        self._hookable_layers = self._build_hookable_layers()
        print(f"Available hookable layers: {', '.join(self._hookable_layers.keys())}")
        if hook_layers is None:
            hook_layers = ['conv4']
        self.set_hook_layers(hook_layers)

    def _build_hookable_layers(self):
        hookable = {}
        for name, module in self.main._modules.items():
            if isinstance(module, nn.Conv2d):
                hookable[name] = module
        # expose numbered conv layers for backward compatibility
        conv_names = [name for name, module in self.main._modules.items() if isinstance(module, nn.Conv2d)]
        for idx, name in enumerate(conv_names, start=1):
            hookable['conv%d' % idx] = self.main._modules[name]
        return hookable

    def set_hook_layers(self, hook_layers):
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []

        if hook_layers is None:
            self.hook_layers = []
            return

        self.hook_layers = list(hook_layers)
        if len(self.hook_layers) == 0:
            return

        missing = [name for name in self.hook_layers if name not in self._hookable_layers]
        if missing:
            available = ', '.join(sorted(self._hookable_layers.keys()))
            raise ValueError('Unknown hook layer(s): %s. Available: %s' % (', '.join(missing), available))

        for name in self.hook_layers:
            module = self._hookable_layers[name]
            self._hook_handles.append(module.register_forward_hook(self._capture_hook(name)))

    def _capture_hook(self, name):
        def hook(_module, _input, output):
            self._hooked[name] = output
        return hook

    def forward(self, input_x):
        self._hooked = {}
        if isinstance(input_x.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input_x, range(self.ngpu))
        else:
            output = self.main(input_x)
        output = output.view(output.size(0), -1)
        if isinstance(output.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.classifier, output, range(self.ngpu))
        else:
            output = self.classifier(output)

        if getattr(self, 'hook_layers', None):
            features = [self._hooked[name] for name in self.hook_layers if name in self._hooked]
        else:
            features = []
        return output.view(-1, 1), features
