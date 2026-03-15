import datetime
import logging
import os
import random
import sys

import torch
import torch.backends.cudnn as cudnn


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
