import os
import sys
import numpy as np
import torchvision.models as models

abspath = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, abspath + '/../')

import torch
from torch import nn

from common.loss import l1_loss, l2_loss
from common.checkpoint import load_model, save_model
from config import opt


class HmrLoss(nn.Module):
    def __init__(self):
        super(HmrLoss, self).__init__()
        # self.smpl = SMPL(opt.smpl_basic_path, opt.smpl_cocoplus_path, smpl_type=opt.smpl_type)
        # self.register_buffer('loss', torch.zeros(8).type(torch.float32))
        # print('finished create smpl module.')

    def forward(self, output, batch):
        # poss_loss, shape_loss = self.loss.zero_()

        ## 1.loss of pose shape
        pose_0_loss = l2_loss(output['pose'][:,0,:], batch['pose'][:, 0,:])
        pose_1_loss = l2_loss(output['pose'][:,1,:], batch['pose'][:, 1,:])

        ## total loss
        loss = opt.pose_weight * (pose_0_loss + pose_1_loss)

        loss_stats = { 'loss': loss.item(),
                       'pose_0': pose_0_loss.item(),
                       'pose_1': pose_1_loss.item()}

        return loss, loss_stats


class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()

        print('start creating sub modules...')
        self.create_sub_modules()
        print('finished create sub modules module.')

    def create_sub_modules(self):
        self.BaseNet = models.resnet50(pretrained=True, progress=True)
        self.fc = nn.Linear(self.BaseNet.fc.out_features, 24*3*2)

    def forward(self, input):
        out = self.BaseNet(input)
        out = self.fc(out).view(-1, 2, 72)

        output = {
            "pose": out
        }
        return output


class ModelWithLoss(nn.Module):
    def __init__(self, model, loss):
        super(ModelWithLoss, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, batch):
        output = self.model(batch['input'])
        loss, loss_states = self.loss(output, batch)
        return output, loss, loss_states