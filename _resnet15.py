'''
from models.LightResnets import ResNet15
from models.heads import SupervisedModel
from torchsummary.torchsummary import summary

backbone = ResNet15()
model = SupervisedModel(backbone, num_labels=35)

print(summary(model, (1,240,240), batch_size=1, device='cpu'))
'''

from models.LambdaResnets_1D import LambdaResNet18
from models.heads import SupervisedModel
from torchsummary.torchsummary import summary

backbone = LambdaResNet18(in_channels=1)
model = SupervisedModel(backbone, num_labels=35)

print(summary(model, (1,16000), batch_size=1, device='cpu'))