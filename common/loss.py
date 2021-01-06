
import os
import sys
import torch


abspath = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, abspath + '/../')


def l1_loss(pred, target):
    loss = torch.abs(pred - target).sum()
    loss = loss / pred.numel()
    return loss


def l2_loss(pred, target):
    loss = torch.sum((pred - target)**2)
    loss = loss / pred.numel()
    return loss


def mask_loss(pred, target):
    out = pred[(((1 - target) + pred) > 1)]
    loss = 0.5 * torch.sum((pred - target)**2) + 1.0 * torch.sum(out**2)
    loss = loss / pred.numel()
    return loss


def part_mask_loss(pred, target):
    out = pred[(((1 - target) + pred) > 1)]
    loss = 0.5 * torch.sum((pred - target)**2) + 1.0 * torch.sum(out**2)
    loss = loss / pred.numel()
    return loss


