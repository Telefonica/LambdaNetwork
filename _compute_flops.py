#from models.LambdaResnets_2D import LambdaResNet18, LambdaResNet50
from models.LambdaResnets_1D import LambdaResNet18, LambdaResNet50
from torchscan import summary
from utils.common_config import get_dataset, get_train_transformations
from utils.config import create_config
from models.heads import SupervisedModel

config_exp = './configs/google_commands/LambdaResnet1D_raw.yml'
config_env = './configs/env.yml'

p = create_config(config_env, config_exp)
train_transforms = get_train_transformations(p)
train_dataset = get_dataset(p, train_transforms, subset="training")

backbone = LambdaResNet18(in_channels=1)

#backbone = LambdaResNet50(in_channels=1)
#backbone = ResNet50(in_channels=1)

#backbone = ResNet15(in_channels=1)

#num_labels = 10
#num_labels = 20
num_labels = 35

model = SupervisedModel(backbone, head='mlp', num_labels=num_labels)
import torch
print(train_dataset[0]['input'].shape)
print(summary(model, train_dataset[0]['input'].shape))