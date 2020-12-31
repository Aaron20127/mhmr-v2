
import os
import sys
import torch


abspath = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, abspath + '/../')


def l1_loss(pred, target):
    loss = torch.abs(pred - target).sum()
    loss = loss / (pred.numel() + 1e-16)
    return loss


def l2_loss(pred, target):
    loss = torch.sum((pred - target)**2)
    loss = loss / (pred.numel() + 1e-16)
    return loss


