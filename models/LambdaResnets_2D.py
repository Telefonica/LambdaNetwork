import torch
import torch.nn as nn
import torch.nn.functional as F
from models.LambdaLayers import LambdaLayer2D as LambdaConv


class LambdaBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(LambdaBlock, self).__init__()
        self.is_last = is_last

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=(3,3), stride=stride, padding=1, bias=False)    
        self.bn1 = nn.BatchNorm2d(planes)
        #self.conv1 = nn.ModuleList([LambdaConv(in_planes, planes)])
        #if stride != 1 or in_planes != self.expansion * planes:
        #    self.conv1.append(nn.AvgPool2d(kernel_size=(3, 3), stride=stride, padding=(1, 1)))
        #self.conv1.append(nn.BatchNorm2d(planes))
        #self.conv1.append(nn.ReLU())
        #self.conv1 = nn.Sequential(*self.conv1)
        
        self.conv2 = nn.ModuleList([LambdaConv(planes, planes)])
        if stride != 1 or in_planes != self.expansion * planes:
            self.conv2.append(nn.AvgPool2d(kernel_size=(3, 3), stride=1, padding=(1, 1)))
        self.conv2.append(nn.BatchNorm2d(planes))
        self.conv2.append(nn.ReLU())
        self.conv2 = nn.Sequential(*self.conv2)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        #out = self.conv1(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class LambdaBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(LambdaBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.ModuleList([LambdaConv(planes, planes)])
        if stride != 1 or in_planes != self.expansion * planes:
            self.conv2.append(nn.AvgPool2d(kernel_size=(3, 3), stride=stride, padding=(1, 1)))
        self.conv2.append(nn.BatchNorm2d(planes))
        self.conv2.append(nn.ReLU())
        self.conv2 = nn.Sequential(*self.conv2)

        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# reference
# https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000, in_channels=3):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0])
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        #self.fc = nn.Sequential(
        #    nn.Dropout(0.3), # All architecture deeper than ResNet-200 dropout_rate: 0.2
        #    nn.Linear(512 * block.expansion, num_classes)
        #)

    def _make_layer(self, block, planes, num_blocks, stride=1):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for idx, stride in enumerate(strides):
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if x.shape[1] != 1:
            x = torch.unsqueeze(x, dim=1)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        #out = self.fc(out)
        return out


class LambdaResNet15_2d(nn.Module):
    def __init__(self, in_channels, n_maps):
        super().__init__()
        n_maps = n_maps
        self.conv0 = nn.Conv2d(in_channels, n_maps, (3, 3), padding=(1, 1), bias=False)
        self.n_layers = n_layers = 14
        dilation = True

        self.convs = [nn.Conv2d(n_maps, n_maps, (3, 3), padding=1, dilation=1, bias=False) for _ in range(n_layers//2)]
        #self.convs = [nn.Conv2d(n_maps, n_maps, (3, 3), padding=int(2 ** (2*i // 3)), dilation=int(2 ** (2*i // 3)), bias=False) for i in range(n_layers//2)]
        
        self.lambdas = [LambdaConv(n_maps, n_maps) for _ in range(n_layers//2)]
        
        '''
        if dilation:
            self.convs = [nn.Conv2d(n_maps, n_maps, (3, 3), padding=int(2 ** (i // 3)), dilation=int(2 ** (i // 3)),
                                    bias=False) for i in range(n_layers)]
        else:
            self.convs = [nn.Conv2d(n_maps, n_maps, (3, 3), padding=1, dilation=1,
                                    bias=False) for _ in range(n_layers)]
        '''
        
        for i, conv in enumerate(self.convs):
            self.add_module("bn{}".format(i + 1), nn.BatchNorm2d(n_maps, affine=False))
            self.add_module("conv{}".format(i + 1), conv)
        
        for i, lambdaLayer in enumerate(self.lambdas):
            self.add_module("bn_lambda{}".format(i + 1), nn.BatchNorm2d(n_maps, affine=False))
            self.add_module("lambda{}".format(i + 1), lambdaLayer)


    def forward(self, audio_signal, length=None):
        x = audio_signal
        #x = audio_signal.unsqueeze(1)
        for i in range(self.n_layers + 1):
            if i == 0:
                y = F.relu(getattr(self, "conv{}".format(i))(x))
                if hasattr(self, "pool"):
                    y = self.pool(y)
                old_x = y
            else:
                # Lambda Layer
                if i % 2 == 0:
                    y = F.relu(getattr(self, "lambda{}".format(i//2))(x))
                # Conv Layer
                else:
                    y = F.relu(getattr(self, "conv{}".format((i+1)//2))(x))
            
            if i > 0 and i % 2 == 0:
                x = y + old_x
                old_x = x
            else:
                x = y
            
            if i > 0:
                # Lambda Layer
                if i % 2 == 0:
                    x = getattr(self, "bn_lambda{}".format(i//2))(x)
                # Conv Layer
                else:
                    x = getattr(self, "bn{}".format((i+1)//2))(x)
        
        x = x.view(x.size(0), x.size(1), -1)  # shape: (batch, feats, o3)
        x = torch.mean(x, 2)
        return x
        #return x.unsqueeze(-2), length




def LambdaResNet18(in_channels=1):
    return {'backbone': ResNet(LambdaBlock, [2, 2, 2, 2], in_channels=in_channels), 'dim': 512}

def LambdaResNet50(in_channels=1):
    return {'backbone': ResNet(LambdaBottleneck, [3, 4, 6, 3], in_channels=in_channels), 'dim': 2048}

def LambdaResNet15(in_channels=1, n_maps=44):
    return {'backbone': LambdaResNet15_2d(in_channels=in_channels, n_maps=n_maps), 'dim': 44}

# reference
# https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325
def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def check_params():

    model = LambdaResNet18()
    print('LambdaResNet18: ', get_n_params(model))

    model = LambdaResNet50()
    print('LambdaResNet50: ', get_n_params(model))

# check_params()
