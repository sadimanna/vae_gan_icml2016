import datetime
import logging
import os
import random

import torch
import torch.backends.cudnn as cudnn


def setup_output_dir(base_outf):
    try:
        os.makedirs(base_outf)
    except OSError:
        pass

    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(base_outf, 'run_%s' % ts)
    try:
        os.makedirs(run_dir)
    except OSError:
        pass

    return run_dir


def setup_logger(run_dir):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)

    logfile = os.path.join(run_dir, 'run.log')
    file_handler = logging.FileHandler(logfile)
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
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
