from __future__ import print_function

import argparse
import torch

from data import build_dataloader
from models import Discriminator, Encoder, Generator, Sampler, weights_init
from train import Trainer
from utils import set_seed, setup_logger, setup_output_dir


def build_parser():
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
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--netG', default='', help='path to netG (to continue training)')
    parser.add_argument('--netD', default='', help='path to netD (to continue training)')
    parser.add_argument('--outf', default='outputs', help='folder to output images and model checkpoints')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--gamma', type=float, default=10, help='weight for reconstruction loss in generator objective')
    parser.add_argument('--kld_wt', type=float, default=0.00025, help='weight for KL divergence loss in encoder objective')
    parser.add_argument('--stopIter', type=int, default=10, help='iteration to stop training if early stopping desired (default is effectively no early stopping)')
    return parser


def main():
    parser = build_parser()
    opt = parser.parse_args()

    run_dir = setup_output_dir(opt.outf)
    logger = setup_logger(run_dir)
    logger.info(opt)
    opt.outf = run_dir

    set_seed(opt, logger)

    if torch.cuda.is_available() and not opt.cuda:
        logger.warning('You have a CUDA device, so you should probably run with --cuda')

    dataloader = build_dataloader(opt)

    nc = 3
    encoder = Encoder(opt.imageSize, opt.nz, opt.ngf, nc)
    sampler = Sampler()
    netG = Generator(opt.nz, opt.ngf, nc, opt.ngpu)

    encoder.apply(weights_init)
    netG.apply(weights_init)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
    logger.info(netG)

    netD = Discriminator(opt.imageSize, opt.ndf, nc, opt.ngpu)
    netD.apply(weights_init)
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    logger.info(netD)

    trainer = Trainer(opt, encoder, sampler, netG, netD, logger)
    trainer.train(dataloader)


if __name__ == '__main__':
    main()
