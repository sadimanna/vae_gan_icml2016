from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from torchvision import models
import torch.nn.functional as F

# we'll need sklearn for TSNE, install if missing


class Sampler(nn.Module):
    def __init__(self, cuda=False):
        super(Sampler, self).__init__()
        self.cuda = cuda

    def forward(self, input):
        mu, logvar = input
        std = logvar.mul(0.5).exp_()
        if self.cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)


class _Encoder(nn.Module):
    def __init__(self, imageSize, nc, ngf, nz):
        super(_Encoder, self).__init__()
        
        n = math.log2(imageSize)
        assert n == round(n), 'imageSize must be a power of 2'
        assert n >= 3, 'imageSize must be at least 8'
        n = int(n)

        # output convolutional layers for mean and logvar
        self.conv1 = nn.Conv2d(ngf * 2 ** (n - 3), nz, 4)
        self.conv2 = nn.Conv2d(ngf * 2 ** (n - 3), nz, 4)

        # build a conventional convolutional encoder (same depth as before)
        self.encoder = nn.Sequential()
        # first block: input -> ngf
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

        # state size now: (ngf*2**(n-3)) x 4 x 4

    def forward(self, input):
        output = self.encoder(input)
        return [self.conv1(output), self.conv2(output)]


class NetG(nn.Module):
    def __init__(self, nz, ngf, nc, ngpu=1):
        super(NetG, self).__init__()
        self.ngpu = ngpu
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
            nn.Tanh(),
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


# parsing and helper functions
import math

def get_dataset(args):
    # build transform
    transform = transforms.Compose([
        transforms.Resize(args.imageSize),
        transforms.CenterCrop(args.imageSize),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    if args.dataset in ['imagenet', 'folder', 'lfw']:
        ds = dset.ImageFolder(root=args.dataroot, transform=transform)
    elif args.dataset == 'lsun':
        ds = dset.LSUN(db_path=args.dataroot, classes=['bedroom_train'], transform=transform)
    elif args.dataset == 'cifar10':
        ds = dset.CIFAR10(root=args.dataroot, train=False, download=True, transform=transform)
    else:
        raise ValueError('Dataset %s not supported in generate script' % args.dataset)
    return ds


def tsne_and_plot(z_vectors, images, outpath):
    from sklearn.manifold import TSNE
    # adjust perplexity to be less than number of samples
    n = z_vectors.shape[0]
    perp = min(30, max(1, n // 3))
    tsne = TSNE(n_components=2, perplexity=perp, random_state=0)
    z2d = tsne.fit_transform(z_vectors)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(z2d[:, 0], z2d[:, 1], s=10, alpha=0.0)  # invisible points, will annotate with images

    for (x, y), img in zip(z2d, images):
        im = OffsetImage(img, zoom=0.5)
        ab = AnnotationBbox(im, (x, y), xycoords='data', frameon=False)
        ax.add_artist(ab)

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='dataset name')
    parser.add_argument('--dataroot', required=True, help='path to dataset')
    parser.add_argument('--imageSize', type=int, default=64, help='height/width of images')
    parser.add_argument('--nz', type=int, default=100, help='size of latent vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--ngpu', type=int, default=1)
    parser.add_argument('--netG', default='', help='path to generator checkpoint (optional)')
    parser.add_argument('--netE', default='', help='path to encoder checkpoint (optional)')
    parser.add_argument('--outf', default='.', help='output folder')
    parser.add_argument('--num', type=int, default=200, help='number of test images to process')
    parser.add_argument('--perturb_std', type=float, default=0.1, help='std of gaussian perturbation added to latent vectors')
    parser.add_argument('--evalBatch', type=int, default=64, help='batch size to use when encoding/generating')
    parser.add_argument('--cuda', action='store_true', help='enable cuda')
    args = parser.parse_args()

    try:
        os.makedirs(args.outf)
    except OSError:
        pass

    # device
    device = torch.device('cuda:0' if args.cuda and torch.cuda.is_available() else 'cpu')

    # models
    ngpu = int(args.ngpu)
    nz = int(args.nz)
    ngf = int(args.ngf)
    ndf = int(args.ndf)
    nc = 3
    encoder = _Encoder(args.imageSize, nc, ngf, nz).to(device)
    netG = NetG(args.nz, args.ngf, nc, args.ngpu).to(device)

    # helper from main for weight init
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
    
    encoder.apply(weights_init)
    netG.apply(weights_init)

    if args.netE:
        encoder.load_state_dict(torch.load(args.netE, map_location=device))
    if args.netG:
        netG.load_state_dict(torch.load(args.netG, map_location=device))
    encoder.eval()
    netG.eval()

    sampler = Sampler(cuda=args.cuda)

    # prepare pretrained ResNet18 for feature extraction
    resnet = models.resnet18(pretrained=True)
    # drop the final fully-connected layer, keep feature extractor
    resnet = nn.Sequential(*list(resnet.children())[:-1])
    resnet.to(device)
    resnet.eval()

    # dataset & loader
    dataset = get_dataset(args)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.evalBatch, shuffle=True, num_workers=2)

    collected = []
    z_list = []
    total = 0
    dataiter = iter(dataloader)
    while total < args.num:
        try:
            real_imgs, _ = next(dataiter)
        except StopIteration:
            break
        real_imgs = real_imgs.to(device)
        bs = real_imgs.size(0)

        with torch.no_grad():
            encoded = encoder(real_imgs)
            mu, logvar = encoded
            z_enc = sampler((mu, logvar))
            noise = torch.randn_like(z_enc) * args.perturb_std
            z_pert = z_enc + noise
            gen_batch = netG(z_pert.view(bs, args.nz, 1, 1))

        collected.append(gen_batch.cpu())
        # compute ResNet features for generated images
        with torch.no_grad():
            # convert to [0,1] and resize to 224
            inp = (gen_batch + 1) / 2
            inp = F.interpolate(inp, size=224, mode='bilinear', align_corners=False)
            # normalize using ImageNet statistics
            mean = torch.tensor([0.485, 0.456, 0.406], device=inp.device).view(1,3,1,1)
            std = torch.tensor([0.229, 0.224, 0.225], device=inp.device).view(1,3,1,1)
            inp = (inp - mean) / std
            feats = resnet(inp)  # shape [bs,512,1,1]
            feats = feats.view(bs, -1).cpu()
        z_list.append(feats)
        total += bs

    if total == 0:
        raise RuntimeError("Dataset provided no images")

    gen_all = torch.cat(collected, 0)[:args.num]
    z_vectors = torch.cat(z_list, 0)[:args.num].numpy()

    # prepare images for plotting (convert tensors in range [-1,1] to numpy [0,1])
    gen_np = gen_all.clamp(-1, 1)
    gen_np = (gen_np + 1) / 2
    gen_imgs = [np.transpose(img.numpy(), (1, 2, 0)) for img in gen_np]

    tsne_path = os.path.join(args.outf, 'tsne_plot.png')
    tsne_and_plot(z_vectors, gen_imgs, tsne_path)

    # also save a grid of generated images from encoded+perturbed vectors
    grid_path = os.path.join(args.outf, 'gen_grid.png')
    vutils.save_image(gen_all, grid_path, normalize=True, nrow=int(math.sqrt(gen_all.size(0))))

    # generate some pure random samples as baseline (batch-wise to avoid OOM)
    rand_chunks = []
    remaining = args.num
    while remaining > 0:
        curr = min(remaining, args.evalBatch)
        with torch.no_grad():
            noise = torch.randn(curr, args.nz, 1, 1, device=device)
            rand_chunks.append(netG(noise).cpu())
        remaining -= curr
    rand_gen = torch.cat(rand_chunks, 0)[:args.num]
    rand_grid = os.path.join(args.outf, 'rand_grid.png')
    vutils.save_image(rand_gen, rand_grid, normalize=True, nrow=int(math.sqrt(args.num)))

    print('visualization written to', args.outf)


if __name__ == '__main__':
    main()
