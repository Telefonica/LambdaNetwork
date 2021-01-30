from models.LightResnets import ResNet15
from models.heads import SupervisedModel
from torchsummary.torchsummary import summary
backbone = ResNet15()
model = SupervisedModel(backbone, num_labels=35)

print(summary(model, (1,240,240), batch_size=1, device='cpu'))