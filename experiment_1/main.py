
import os
import sys
import torch
from tqdm import tqdm

abspath = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, abspath + '/../')

from config import opt
from common.log import Logger
from train import HMRTrainer


def main():
    ## 1. basic
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)

    torch.backends.cudnn.benchmark = opt.cuda_benchmark
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus

    ## 2. log
    opt.logger = Logger(opt, abspath)

    ## 3. train
    trainer = HMRTrainer(opt.logger)

    if opt.train:
        trainer.train()
    else:
        trainer.val()

if __name__ == '__main__':
    main()



