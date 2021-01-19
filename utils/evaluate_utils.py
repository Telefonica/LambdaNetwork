"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import numpy as np
import torch
import torch.nn.functional as F
from utils.common_config import get_feature_dimensions_backbone
from utils.utils import AverageMeter, confusion_matrix
from data.custom_dataset import NeighborsDataset
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
from losses.losses import entropy


@torch.no_grad()
def contrastive_evaluate(val_loader, model, memory_bank):
    top1 = AverageMeter('Acc@1', ':6.2f')
    model.eval()

    for batch in val_loader:
        images = batch['image'].cuda(non_blocking=True)

        output = model(images)
        output = memory_bank.weighted_knn(output) 

        acc1 = 100*torch.mean(torch.eq(output, target).float())
        top1.update(acc1.item(), images.size(0))

    return top1.avg
