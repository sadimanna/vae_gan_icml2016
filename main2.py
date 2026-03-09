from __future__ import print_function
import argparse
import os
import random
import math
import re
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import logging
import datetime


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class _Sampler(nn.Module):
    def __init__(self):
        super(_Sampler, self).__init__()
        
    def forward(self,input_x):
        mu = input_x[0]
        logvar = input_x[1]
        
        std = logvar.mul(0.5).exp_() #calculate the STDEV
        eps = torch.randn(std.size(), device=std.device)
        eps = Variable(eps)
        return eps.mul(std).add_(mu) 


class _Encoder(nn.Module):
    def __init__(self,imageSize):
        super(_Encoder, self).__init__()
        
        n = math.log2(imageSize)
        
        assert n==round(n),'imageSize must be a power of 2'
        assert n>=3,'imageSize must be at least 8'
        n=int(n)


        self.conv1 = nn.Conv2d(ngf * 2**(n-3), nz, 4)
        self.conv2 = nn.Conv2d(ngf * 2**(n-3), nz, 4)

        # build a conventional convolutional encoder (same depth as before)
        self.encoder = nn.Sequential()
        # first block: input_x -> ngf
        self.encoder.add_module('conv1', nn.Conv2d(nc, ngf, 4, 2, 1, bias=False))
        self.encoder.add_module('lrelu1', nn.LeakyReLU(0.2, inplace=True))
        # subsequent blocks doubling channels each time
        for i in range(n-3):
            in_ch = ngf * 2**i
            out_ch = ngf * 2**(i+1)
            idx = i + 2
            self.encoder.add_module('conv%d' % idx, nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False))
            self.encoder.add_module('bn%d' % idx, nn.BatchNorm2d(out_ch))
            self.encoder.add_module('lrelu%d' % idx, nn.LeakyReLU(0.2, inplace=True))

        # state size. (ngf*2**(n-3)) x 4 x 4

    def forward(self,input_x):
        output = self.encoder(input_x)
        return [self.conv1(output),self.conv2(output)]


class _netG(nn.Module):
    def __init__(self, nz, ngf, nc, ngpu=1):
        super(_netG, self).__init__()
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




class _netD(nn.Module):
    def __init__(self, imageSize, ngpu):
        super(_netD, self).__init__()
        self.ngpu = ngpu
        n = math.log2(imageSize)
        
        assert n==round(n),'imageSize must be a power of 2'
        assert n>=3,'imageSize must be at least 8'
        n=int(n)
        # build discriminator dynamically so it supports multiple image sizes
        self.main = nn.Sequential()
        # input_x conv
        self.main.add_module('input-conv', nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        self.main.add_module('input-lrelu', nn.LeakyReLU(0.2, inplace=True))
        # add pyramid blocks based on image size
        for i in range(n-3):
            in_ch = ndf * 2 ** i
            out_ch = ndf * 2 ** (i + 1)
            self.main.add_module('conv%d' % (i), nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False))
            self.main.add_module('bn%d' % (i), nn.BatchNorm2d(out_ch))
            self.main.add_module('lrelu%d' % (i), nn.LeakyReLU(0.2, inplace=True))

        # final output conv to single score (kernel should match remaining spatial size)
        self.main.add_module('output-conv', nn.Conv2d(ndf * 2 ** (n-3), 1, 4, 1, 0, bias=False))
        self.main.add_module('output-sigmoid', nn.Sigmoid())

        # Dictionary to store the activations
        self.activations = []

        # 2. Define the hook function
        def save_activations(module, input, output):
            # Detach the output from the computational graph to avoid issues
            # and store it in the dictionary
            self.activations.append(output)
            # You can also use the hook to print information
            # print(f"Hook triggered! Output shape: {output.shape}")
        
        for name, layer in self.main.named_modules():
            if isinstance(layer, nn.Conv2d) and 'conv1' in name:
                layer.register_forward_hook(save_activations)

    def forward(self, input_x):
        if isinstance(input_x.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input_x, range(self.ngpu))
        else:
            output = self.main(input_x)

        return self.activations, output.view(-1, 1)

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)


    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw ')
    parser.add_argument('--dataroot', required=True, help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--batchSize', type=int, default=64, help='input_x batch size')
    parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input_x image to network')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
    parser.add_argument('--saveInt', type=int, default=5, help='number of epochs between checkpoints')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.5') # Why 0.5??
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--outf', default='outputs', help='folder to output images and model checkpoints')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--gamma', type=float, default=10, help='weight for reconstruction loss in generator objective')
    parser.add_argument('--kld_wt', type=float, default=0.00025, help='weight for KL divergence loss in encoder objective')
    parser.add_argument('--stopIter', type=int, default=10, help='iteration to stop training if early stopping desired (default is effectively no early stopping)')

    opt = parser.parse_args()
    # ensure base output directory exists
    try:
        os.makedirs(opt.outf)
    except OSError:
        pass
    # create unique run subdirectory
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(opt.outf, f'run_{ts}')
    try:
        os.makedirs(run_dir)
    except OSError:
        pass
    # reconfigure logger to write to run folder
    logfile = os.path.join(run_dir, 'run.log')
    file_handler = logging.FileHandler(logfile)
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info(opt)
    # redirect outputs to run_dir
    opt.outf = run_dir

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    logger.info("Random Seed: %s", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not opt.cuda:
        logger.warning("You have a CUDA device, so you should probably run with --cuda")

    if opt.dataset in ['imagenet', 'folder', 'lfw']:
        # folder dataset
        dataset = dset.ImageFolder(root=opt.dataroot,
                                transform=transforms.Compose([
                                    transforms.Resize(opt.imageSize),
                                    transforms.CenterCrop(opt.imageSize),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]))
    elif opt.dataset == 'lsun':
        dataset = dset.LSUN(db_path=opt.dataroot, classes=['bedroom_train'],
                            transform=transforms.Compose([
                                transforms.Resize(opt.imageSize),
                                transforms.CenterCrop(opt.imageSize),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
    elif opt.dataset == 'cifar10':
        dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                            transform=transforms.Compose([
                                transforms.Resize(opt.imageSize),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ])
        )
    assert dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                            shuffle=True, num_workers=int(opt.workers))

    ngpu = int(opt.ngpu)
    nz = int(opt.nz)
    ngf = int(opt.ngf)
    ndf = int(opt.ndf)
    nc = 3

    patience = opt.stopIter
    prevDLoss = None
    prevGLoss = None

    # decoupled encoder + sampler and DCGAN-style generator
    encoder = _Encoder(opt.imageSize)
    sampler = _Sampler()
    netG = _netG(nz, ngf, nc, ngpu)
    # initialize weights
    encoder.apply(weights_init)
    netG.apply(weights_init)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
    logger.info(netG)

    netD = _netD(opt.imageSize,ngpu)
    netD.apply(weights_init)
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    logger.info(netD)
            

    criterion = nn.BCELoss()
    MSECriterion = nn.MSELoss()

    input_x = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
    noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
    fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
    label = torch.FloatTensor(opt.batchSize)
    real_label = 1
    fake_label = 0

    if opt.cuda:
        netD.cuda()
        netG.cuda()
        encoder.cuda()
        sampler.cuda()
        criterion.cuda()
        MSECriterion.cuda()
        input_x, label = input_x.cuda(), label.cuda()
        noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

    input_x = Variable(input_x)
    label = Variable(label)
    noise = Variable(noise)
    fixed_noise = Variable(fixed_noise)

    # setup optimizers: separate for encoder and generator
    optimizerD = optim.RMSprop(netD.parameters(), lr=opt.lr) #, betas=(opt.beta1, 0.999))
    optimizerEnc = optim.RMSprop(encoder.parameters(), lr=opt.lr) #, betas=(opt.beta1, 0.999))
    optimizerG = optim.RMSprop(netG.parameters(), lr=opt.lr) #, betas=(opt.beta1, 0.999))

    # image windows replaced by saved image files in `opt.outf`

    for epoch in range(opt.niter):
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(Dec(z))) + log(1-D(Dec(Enc(x))))
            ###########################
            # train with real
            netD.zero_grad()
            real_cpu, _ = data
            batch_size = real_cpu.size(0)
            input_x.data.resize_(real_cpu.size()).copy_(real_cpu)
            label.data.resize_(real_cpu.size(0)).fill_(real_label)

            # Dis(x)
            _, output = netD(input_x)
            errD_real = criterion(output, label.view(-1, 1))
            # errD_real.backward()
            D_x = output.data.mean()
            # train with fake - Dis(Dec(z))
            noise.data.resize_(batch_size, nz, 1, 1)
            noise.data.normal_(0, 1)
            gen = netG(noise)
            label.data.fill_(fake_label)
            _, output = netD(gen.detach())
            errD_fake = criterion(output, label.view(-1, 1))
            # errD_fake.backward()
            D_G_z1 = output.data.mean()
            # train with reconstructed (encoded->sampled->decoded) -- Dis(Dec(Enc(x)))
            encoded = encoder(input_x)
            sampled = sampler(encoded)
            rec = netG(sampled) # ------------------------> Reconstruction
            _, output = netD(rec.detach())
            errD_rec = criterion(output, label.view(-1, 1))  # label is already fake_label -- right
            # errD_rec.backward()
            D_G_z_rec = output.data.mean()
            errD = errD_real + errD_fake + errD_rec
            errD.backward(retain_graph=False)
            optimizerD.step()
            ############################
            # (2) Update Encoder network: minimize KL divergence + reconstruction error
            ###########################
            
            encoder.zero_grad()
            # netG.zero_grad()
            encoded = encoder(input_x)
            mu = encoded[0]
            logvar = encoded[1]
            # KLD Loss
            KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
            KLD_loss = torch.sum(KLD_element).mul_(-0.5)
            #########################
            sampled = sampler(encoded)
            rec = netG(sampled)
            # Disl_llike
            acts_r = netD(rec)[0]
            acts_x = netD(input_x)[0]
            MSEerr = MSECriterion(rec, input_x) # acts_r[-1], acts_x[-1])
            # Total error
            VAEerr = opt.kld_wt * KLD_loss + 5.0 * MSEerr
            VAEerr.backward()
            optimizerEnc.step()
            # optimizerG.step()

            ############################
            # (3) Update G network: maximize log(D(G(z))) + reconstruction loss
            ###########################

            label.data.fill_(real_label)  # fake labels are real for generator cost
            netG.zero_grad()
            # Dis(x)
            acts_x, output = netD(input_x)
            errG_real = criterion(output, label.view(-1, 1))
            # errG_real.backward()
            G_x = output.data.mean()
            # train with fake - Dis(Dec(z))
            noise.data.resize_(batch_size, nz, 1, 1)
            noise.data.normal_(0, 1)
            gen = netG(noise)
            label.data.fill_(fake_label)
            _, output = netD(gen.detach())
            errG_fake = criterion(output, label.view(-1, 1))
            # errD_fake.backward()
            D_G_z1 = output.data.mean()
            # reconstruct via encoder->sampler->generator
            encoded = encoder(input_x)
            sampled = sampler(encoded)
            rec = netG(sampled)
            acts_r, output = netD(rec)
            errG_adv = criterion(output, label.view(-1, 1))
            gan_loss = errG_adv + errG_fake + errG_real
            # add reconstruction loss to generator training
            MSEerr_G = MSECriterion(rec, input_x) # acts_r[-1], acts_x[-1])
            errG =  -gan_loss + opt.gamma * MSEerr_G
            errG.backward()
            D_G_z2 = output.data.mean()
            optimizerG.step()

            if i % 500 == 0:
                # save generated batch to file
                vutils.save_image(gen, os.path.join(opt.outf, 'gen_epoch%d_iter%d.png' % (epoch, i)), normalize=True, value_range=(-1, 1))
                # save reconstruction batch to file
                vutils.save_image(rec, os.path.join(opt.outf, 'rec_epoch%d_iter%d.png' % (epoch, i)), normalize=True, value_range=(-1, 1))
                logger.info('[%d/%d][%d/%d] Loss_VAE: %.4f Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f D(Dec(Enc(x))): %.4f',
                    epoch+1, opt.niter, i, len(dataloader), VAEerr.item(), errD.item(), errG.item(), D_x, D_G_z1, D_G_z_rec)
                
                if prevDLoss is not None and prevGLoss is not None:
                    if errD.item() > prevDLoss and abs(errG.item()) > prevGLoss:
                        patience -= 1
                        logger.info('No improvement in D or G loss, reducing patience to %d', patience)
                        if patience <= 0:
                            logger.info('Early stopping at epoch %d, iteration %d', epoch+1, i)
                            break
                    else:
                        patience = opt.stopIter
                        prevDLoss = errD.item()
                        prevGLoss = abs(errG.item())
                else:
                    prevDLoss = errD.item()
                    prevGLoss = abs(errG.item())
        if patience <= 0:
            logger.info('Early stopping at epoch %d, iteration %d', epoch+1, i)
            break

        if (epoch+1)%opt.saveInt == 0 and epoch!=0:
            torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch+1))
            torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch+1))
            torch.save(encoder.state_dict(), '%s/encoder_epoch_%d.pth' % (opt.outf, epoch+1))
