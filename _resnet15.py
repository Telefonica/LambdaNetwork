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
from models.Resnets_2D import ResNet15

model = ResNet15(in_channels=1)
backbone = model['backbone']
input = torch.zeros([3, 100, 40])
print(backbone(input).shape)