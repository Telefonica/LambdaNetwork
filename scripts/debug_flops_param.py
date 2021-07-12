import torch
from models.Resnet import ResNet15
from models.TCResnet import TCResnet14
from models.LambdaResnet import LambdaResNet18
from models.heads import ClassificationModel
from torchscan import summary

mel_input = torch.zeros([1, 40, 100])
in_channels_2d = mel_input.shape[0]
in_channels_1d = mel_input.shape[1]

res15 = ClassificationModel(
    ResNet15(in_channels=in_channels_2d, n_maps=45),
    num_labels=10,
    head='linear')

tcres14 = ClassificationModel(
    TCResnet14(in_channels=in_channels_1d, k=1),
    num_labels=10,
    head='linear')

lambdaRes18 = ClassificationModel(
    LambdaResNet18(in_channels=in_channels_1d, k=1),
    num_labels=10,
    head='linear')

lambdaRes18_2 = ClassificationModel(
    LambdaResNet18(in_channels=in_channels_1d, k=2),
    num_labels=10,
    head='linear')

print("------- RESNET 15 (2D) -------")
print(summary(res15, mel_input.shape))

print("------- TC RESNET 14 (1D) -------")
print(summary(tcres14, mel_input.shape))

print("------- LAMBDARESNET 18 (1D) -------")
print(summary(lambdaRes18, mel_input.shape))

print("------- LAMBDARESNET 18_2 (1D) -------")
print(summary(lambdaRes18_2, mel_input.shape))