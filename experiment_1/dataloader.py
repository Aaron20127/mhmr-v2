
import os
import sys

abspath = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, abspath + '/../')

import torch
from torch.utils.data import Dataset, DataLoader
from .opts import opt
from .dataset import DatasetTwoPerson


def dataloader_train():
    data_dir = os.path.join(abspath, 'data_prepare', 'dataset')

    data = DatasetTwoPerson(
        data_dir=data_dir,
        split='train'
    )

    loader = DataLoader(data, batch_size=opt.batch_size,
                              shuffle=True)

    return loader


def dataloader_val():
    data_dir = os.path.join(abspath, 'data_prepare', 'dataset')

    data = DatasetTwoPerson(
        data_dir=data_dir,
        split='val'
    )

    loader = DataLoader(data, batch_size=opt.batch_size,
                              num_workers=opt.num_workers,
                              shuffle=True)

    return loader


def dataloader_test():
    return None


def dataloader():
    return dataloader_train(), dataloader_val(), dataloader_test()