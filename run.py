from __future__ import print_function

import argparse
import os
import torch

from data import build_dataloader
from models import Discriminator, Encoder, Generator, Sampler, weights_init
from train import Trainer
from utils import set_seed, setup_logger, setup_output_dir

def print_config_tree(d, indent=0):
    for i, (k, v) in enumerate(d.items()):
        prefix = "│   " * indent
        connector = "├── "

        if isinstance(v, dict):
            print(f"{prefix}{connector}{k}")
            print_config_tree(v, indent + 1)
        else:
            print(f"{prefix}{connector}{k}: {v}")


def pretty_print_args(args):
    config = vars(args)

    print("\nCONFIG")
    print_config_tree(config)
    print()

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw ')
    parser.add_argument('--dataroot', required=True, help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--batchSize', type=int, default=64, help='input_x batch size')
    parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input_x image to network')
    parser.add_argument('--nz', type=int, default=128, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--niter', type=int, default=30, help='number of epochs to train for')
    parser.add_argument('--saveInt', type=int, default=10, help='number of epochs between checkpoints')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--lr_decay_enc', type=float, default=0.85, help='learning rate decay factor for encoder')
    parser.add_argument('--lr_decay_dec', type=float, default=0.85, help='learning rate decay factor for decoder')
    parser.add_argument('--lr_decay_dis', type=float, default=0.85, help='learning rate decay factor for discriminator')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--netG', default='', help='path to netG (to continue training)')
    parser.add_argument('--netD', default='', help='path to netD (to continue training)')
    parser.add_argument('--outf', default='outputs', help='folder to output images and model checkpoints')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--gamma', type=float, default=0.00001, help='weight for reconstruction loss in generator objective')
    parser.add_argument('--kld_wt', type=float, default=0.00025, help='weight for KL divergence loss in encoder objective')
    parser.add_argument('--stopIter', type=int, default=10, help='iteration to stop training if early stopping desired (default is effectively no early stopping)')
    parser.add_argument('--hook_layers', nargs='+', type=str, default='conv3', help='comma-separated discriminator conv layer names for hooks (empty to disable)')
    parser.add_argument('--eval_samples', type=int, default=5000, help='number of samples for evaluation metrics (0 to skip)')
    parser.add_argument('--equillibrium', type=float, default=0.68, help='equilibrium hyperparameter for BEGAN-like training')
    parser.add_argument('--margin', type=float, default=0.35, help='margin for equillibrium to decide whether to pause generator or discriminator training')
    parser.add_argument('--decay_equillibrium', type=float, default=1.0, help='equillibrium decay factor')
    parser.add_argument('--decay_margin', type=float, default=1.0, help='margin decay factor')
    parser.add_argument('--lambda_mse', type=float, default=1e-6, help='weight for MSE loss in discriminator objective')
    parser.add_argument('--decay_mse', type=float, default=1.0, help='decay factor for MSE loss weight')
    return parser


def format_run_tag(opt):
    tag = f"nz{opt.nz}_gm{opt.gamma}_kl{opt.kld_wt}_ep{opt.niter}_bs{opt.batchSize}_lr{opt.lr}"
    return tag.replace('.', 'p')


def main():
    parser = build_parser()
    opt = parser.parse_args()

    pretty_print_args(opt)

    dataset_tag = f"{opt.dataset}_{format_run_tag(opt)}"
    run_dir = setup_output_dir(opt.outf, run_format='%d:%m:%Y_%H:%M:%S', dataset=dataset_tag)
    print(f"Outputs will be saved in: {os.path.basename(run_dir)}")
    logger = setup_logger(run_dir)
    logger.info(opt)
    opt.outf = run_dir

    mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    if opt.cuda:
        if torch.cuda.is_available():
            opt.device = torch.device('cuda')
        elif mps_available:
            opt.device = torch.device('mps')
            logger.warning('CUDA requested but not available; using MPS instead.')
        else:
            opt.device = torch.device('cpu')
            logger.warning('CUDA requested but not available; using CPU.')
    else:
        if mps_available:
            opt.device = torch.device('mps')
            logger.info('Using MPS because CUDA is not available.')
        else:
            opt.device = torch.device('cpu')
            if torch.cuda.is_available():
                logger.warning('You have a CUDA device, so you should probably run with --cuda')
    opt.cuda = opt.device.type == 'cuda'

    set_seed(opt, logger)

    dataloader = build_dataloader(opt)

    nc = 3
    encoder = Encoder(opt.imageSize, opt.nz, opt.ngf, nc)
    sampler = Sampler()
    netG = Generator(opt.nz, opt.ngf, nc, opt.ngpu, image_size=opt.imageSize)

    encoder.apply(weights_init)
    netG.apply(weights_init)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
    logger.info(netG)

    hook_layers = []
    logger.info(f"Requested hook layers: {opt.hook_layers}")
    raw_hooks = opt.hook_layers
    if isinstance(raw_hooks, (list, tuple)):
        hook_layers = [str(name).strip() for name in raw_hooks if str(name).strip()]
    else:
        raw_hooks = str(raw_hooks).strip()
        if raw_hooks and raw_hooks.lower() not in ['none', 'null']:
            hook_layers = [name.strip() for name in raw_hooks.split(',') if name.strip()]

    netD = Discriminator(opt.imageSize, opt.ndf, nc, opt.ngpu, hook_layers=hook_layers)
    netD.apply(weights_init)
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    logger.info(netD)

    trainer = Trainer(opt, encoder, sampler, netG, netD, logger)
    trainer.train(dataloader)


if __name__ == '__main__':
    main()
