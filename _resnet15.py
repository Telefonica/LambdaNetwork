'''
from models.LightResnets import ResNet15
from models.heads import SupervisedModel
from torchsummary.torchsummary import summary

backbone = ResNet15()
model = SupervisedModel(backbone, num_labels=35)

print(summary(model, (1,240,240), batch_size=1, device='cpu'))
'''

'''
from models.LambdaResnets_1D import LambdaResNet18
from models.heads import SupervisedModel
#from torchsummary.torchsummary import summary
from torchscan import summary

backbone = LambdaResNet18(in_channels=1)
model = SupervisedModel(backbone, num_labels=35)

#print(summary(model, (1,16000), batch_size=1, device='cpu'))
print(summary(model, (1,16000)))
'''

'''
import torch
from models.LambdaResnets_2D import LambdaResNet18
from models.heads import SupervisedModel
backbone = LambdaResNet18(in_channels=1)
model = SupervisedModel(backbone, num_labels=35)
inp = torch.zeros([1, 1, 100, 100])
print(model(inp))
'''

'''
import torch
from models.LambdaResnets_1D import LambdaResNet18
from models.heads import SupervisedModel
input = torch.zeros([1, 100, 40])
in_channels = input.shape[1]
backbone = LambdaResNet18(in_channels=in_channels)
model = SupervisedModel(backbone, num_labels=35)
print(model(input))
'''

import torch
from models.Resnets_2D import ResNet15 as Res_2d
from models.Resnets_1D import ResNet15 as Res_1d
from models.LambdaResnets_2D import LambdaResNet15 as LRes2d
from models.LambdaResnets_1D import LambdaResNet15 as LRes1d
from models.heads import SupervisedModel
from torchscan import summary

input = torch.zeros([1, 1, 40, 100])

res15_2d = Res_2d(in_channels=1, n_maps=44)['backbone']
res15_1d = Res_1d(in_channels=40, n_maps=44)['backbone']
lambdaRes15_2d = LRes2d(in_channels=1, n_maps=48)['backbone']
lambdaRes15_1d = LRes1d(in_channels=40, n_maps=48)['backbone']


#model = SupervisedModel(lambdaRes15_1d(in_channels=40, n_maps=44), head='linear')

print(lambdaRes15_1d)

print("----- RESNET 15 ------")
print(summary(res15_2d, (1,40,101)))

print("----- RESNET 15 1D ------")
print(summary(res15_1d, (1,40,101)))

print("----- LAMBDARESNET 15 2D ------")
print(summary(lambdaRes15_2d, (1,40,101)))

print("----- LAMBDARESNET 15 1D ------")
print(summary(lambdaRes15_1d, (1,40,101)))

import sys
sys.exit()

print('Resnet 15 2D {}'.format(res15(input).shape))
print('Lambda Resnet 15 2D {}'.format(lambdaRes15_2d(input).shape))
print('Lambda Resnet 15 1D {}'.format(lambdaRes15_1d(input).shape))