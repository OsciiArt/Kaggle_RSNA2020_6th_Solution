import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
from math import cos, pi
import numpy as np

def cycle(iterable):
    """
    dataloaderをiteratorに変換
    :param iterable:
    :return:
    """
    while True:
        for x in iterable:
            yield x


class CosineLR(_LRScheduler):
    """SGD with cosine annealing.
    """

    def __init__(self, optimizer, step_size_min=1e-5, t0=100, tmult=2, curr_epoch=-1, last_epoch=-1):
        self.step_size_min = step_size_min
        self.t0 = t0
        self.tmult = tmult
        self.epochs_since_restart = curr_epoch
        super(CosineLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        self.epochs_since_restart += 1

        if self.epochs_since_restart > self.t0:
            self.t0 *= self.tmult
            self.epochs_since_restart = 0

        lrs = [self.step_size_min + (
                0.5 * (base_lr - self.step_size_min) * (1 + cos(self.epochs_since_restart * pi / self.t0)))
               for base_lr in self.base_lrs]

        return lrs


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        

class Bottleneck(nn.Module):
    def __init__(self, input_size):
        super(Bottleneck, self).__init__()
        self.lin1 = nn.Linear(input_size, input_size//2)
        self.bn1 = nn.BatchNorm1d(input_size//2)
        self.relu = nn.ReLU()
        self.do = nn.Dropout(p=0.5)
        self.lin2 = nn.Linear(input_size//2, input_size)
        self.bn2 = nn.BatchNorm1d(input_size)
        
    def forward(self, input):
        x = self.lin1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.do(x)
        x = self.lin2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.do(x)
        
        x = x + input

        return x

class GateBlock(nn.Module):
    def __init__(self, input_size, output_size, activate='sigmoid'):
        super(GateBlock, self).__init__()
        self.lin1 = nn.Linear(input_size, output_size)
        self.bn1 = nn.BatchNorm1d(output_size)
        self.lin2 = nn.Linear(input_size, output_size)
        self.bn2 = nn.BatchNorm1d(output_size)
        if activate=='sigmoid':
            self.activate = nn.Sigmoid()
        elif activate=='softmax':
            self.activate = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        self.do = nn.Dropout(p=0.5)
        
    def forward(self, input):
        x1 = self.lin1(input)
        x1 = self.bn1(x1)
        x2 = self.lin2(input)
        x2 = self.bn2(x2)
        x2 = self.activate(x2)
        x = x1 * x2
        x = self.do(x)

        return x


def nnBlock(input_size, output_size):
    x = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.BatchNorm1d(output_size),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )
    return x
