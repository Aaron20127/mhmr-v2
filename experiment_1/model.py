import os
import sys
import numpy as np
import torchvision.models as models

abspath = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, abspath + '/../')

import torch
from torch import nn

from common.loss import pose_l2_loss, shape_l2_loss
from common.checkpoint import load_model, save_model
from .opts import opt


class HmrLoss(nn.Module):
    def __init__(self):
        super(HmrLoss, self).__init__()
        # self.smpl = SMPL(opt.smpl_basic_path, opt.smpl_cocoplus_path, smpl_type=opt.smpl_type)
        self.register_buffer('loss', torch.zeros(8).type(torch.float32))
        # print('finished create smpl module.')

    def forward(self, output, batch):
        # poss_loss, shape_loss = self.loss.zero_()

        ## 1.loss of pose shape
        pose_loss = pose_l2_loss(output['pose'], batch['pose'])
        shape_loss = shape_l2_loss(output['shape'], batch['shape'])

        ## total loss
        loss = opt.pose_weight * pose_loss + \
               opt.shape_weight * shape_loss


        loss_stats = {'loss': loss,
                      'pose': pose_loss,
                      'shape': shape_loss}

        return loss, loss_stats



class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()

        print('start creating sub modules...')
        self._create_sub_modules()
        print('finished create sub modules module.')

    def _create_sub_modules(self):
        self.BaseNet = models.resnet50(pretrained=True, progress=True)
        self.fc = nn.Linear(BaseNet.fc.out_features, 24*3*2)

    def forward(self, input):
        out = self.BaseNet(input)
        out = self.fc(out).view(-1, 2, 72)
        return out



class ModelWithLoss(nn.Module):
    def __init__(self, model, loss):
        super(ModelWithLoss, self).__init__()
        self.model = model
        self.loss = loss


    def forward(self, batch):
        outputs = self.model(batch['input'])
        loss, loss_states = self.loss(outputs, batch)
        return outputs, loss, loss_states