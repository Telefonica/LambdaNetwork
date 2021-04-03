from models.new_lambda import LambdaResNet15
from models.TC_Resnet import TCResnet14
from models.Resnets_2D import ResNet15
import torch
from models.heads import SupervisedModel

from torchscan import summary


lambdaRes15_1d = LambdaResNet15(in_channels=40)['backbone']
TCResnet_1d = TCResnet14(in_channels=40)['backbone']


lres = SupervisedModel(backbone=LambdaResNet15(in_channels=40, k=1), num_labels=10)
lres_2 = SupervisedModel(backbone=LambdaResNet15(in_channels=40, k=2), num_labels=10)
res = SupervisedModel(backbone=TCResnet14(in_channels=40), num_labels=10)
resnet15 = SupervisedModel(backbone=ResNet15(in_channels=1), num_labels=10)

input = torch.zeros([1, 1, 40, 100])

print("----- LAMBDARESNET 15 1D ------")
print(summary(lres, (1,40,101)))

print("----- LAMBDARESNET 15 1D ------")
print(summary(lres_2, (1,40,101)))

print("----- RESNET 14 1D ------")
print(summary(res, (1,40,101)))


print("----- RESNET 15 2D ------")
print(summary(resnet15, (1,40,101)))