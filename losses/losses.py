import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLoss(nn.Module):
    def __init__(self, weight=None):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, input, target, weight=None, reduction=None):
        return F.cross_entropy(input, target, weight=weight)