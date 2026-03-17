import datetime
import logging
import os
import random
import sys

import torch
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import _LRScheduler


def setup_output_dir(base_outf, run_format=None, dataset=None):
    try:
        os.makedirs(base_outf)
    except OSError:
        pass

    if run_format is None:
        run_format = '%Y%m%d_%H%M%S'
    ts = datetime.datetime.now().strftime(run_format).replace(':','')
    if dataset is not None:
        ts = '_'.join([dataset, ts])
    run_dir = os.path.join(base_outf, 'run_%s' % ts)
    try:
        os.makedirs(run_dir)
    except OSError:
        pass

    return run_dir


def setup_logger(run_dir):
    logger = logging.getLogger("vae_gan")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')

        stream_handler = logging.StreamHandler(stream=sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        logfile = os.path.join(run_dir, 'run.log')
        file_handler = logging.FileHandler(logfile)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def set_seed(opt, logger):
    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    logger.info("Random Seed: %s", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)

    cudnn.benchmark = True


class CosineWarmupLR(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=0.0, last_epoch=-1):
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if total_steps <= 0:
            raise ValueError("total_steps must be > 0")
        self.warmup_steps = int(warmup_steps)
        self.total_steps = int(total_steps)
        self.min_lr = float(min_lr)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        lrs = []
        for base_lr in self.base_lrs:
            if self.warmup_steps > 0 and step <= self.warmup_steps:
                lr = base_lr * step / float(self.warmup_steps)
            else:
                progress = min(max(step - self.warmup_steps, 0), self.total_steps - self.warmup_steps)
                denom = max(self.total_steps - self.warmup_steps, 1)
                cosine = 0.5 * (1.0 + torch.cos(torch.tensor(progress / denom * 3.141592653589793)))
                lr = self.min_lr + (base_lr - self.min_lr) * cosine.item()
            lrs.append(lr)
        return lrs
