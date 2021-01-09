
import os
import sys
import torch
import torch.nn as nn

abspath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(abspath + "/../data/vposer_v1_0")

from vposer_smpl import VPoser


class PosePrior(nn.Module):
    def __init__(self):
        super(PosePrior, self).__init__()

        para_dict = torch.load(abspath + "/../data/vposer_v1_0/snapshots/TR00_E096.pt",
                               map_location=torch.device('cpu'))
        model = VPoser()
        model.load_state_dict(para_dict)

        self.model = model

    def forward(self, joints_3d):
        """
        :param joints_3d (tensor, Nx1xn_jointsx3)
        :return: mean (normal mean)
                 std (normal std)
        """
        mean, std = self.normal_distribution(joints_3d)
        return mean, std

