import torch
import torch.nn as nn
import torch.nn.functional as F
from models.LambdaLayers import LambdaLayer1D as LambdaConv


class LambdaBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(LambdaBlock, self).__init__()
        self.is_last = is_last

        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        
        self.conv2 = nn.ModuleList([LambdaConv(planes, planes)])
        
        if stride != 1 or in_planes != self.expansion * planes:
            self.conv2.append(nn.AvgPool1d(kernel_size=3, stride=1, padding=1))
        
        self.conv2.append(nn.BatchNorm1d(planes))
        self.conv2.append(nn.ReLU())
        self.conv2 = nn.Sequential(*self.conv2)
        
        self.shortcut = nn.Sequential()
        
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class LambdaResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channels, k=1):
        super(LambdaResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=3, stride=1, bias=False)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # There are 4 layers in resnet, each layer has it's number of blocks
        self.layer1 = self._make_layer(block, int(24*k), num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, int(36*k), num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, int(48*k), num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, int(60*k), num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def _make_layer(self, block, planes, num_blocks, stride=1):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for idx, stride in enumerate(strides):
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        
        return nn.Sequential(*layers)

    def forward(self, x):
        # In case the MEL has a 1 channel
        if x.shape[1] == 1 and len(x.shape) > 3:
            x = torch.squeeze(x, dim=1)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out


def LambdaResNet18(in_channels, k=1):
    return {'backbone': LambdaResNet(LambdaBlock, [2, 2, 2, 2], in_channels=in_channels, k=k), 'dim': 60*k}