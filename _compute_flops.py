from models.Resnets import ResNet18, ResNet50
from models.LightResnets import ResNet15
from models.LambdaResnets import LambdaResNet18, LambdaResNet50
from torchscan import summary
from utils.common_config import get_dataset, get_train_transformations
from utils.config import create_config
from models.heads import SupervisedModel

config_exp = './configs/lambdaResnet2D_supervised.yml'
config_env = './configs/env.yml'

p = create_config(config_env, config_exp)
train_transforms = get_train_transformations(p)
train_dataset = get_dataset(p, train_transforms, to_augmented_dataset=False, subset="training")

#backbone = LambdaResNet18(in_channels=1)
backbone = ResNet18(in_channels=1)

#backbone = LambdaResNet50(in_channels=1)
#backbone = ResNet50(in_channels=1)

#backbone = ResNet15(in_channels=1)

#num_labels = 10
#num_labels = 20
num_labels = 35

model = SupervisedModel(backbone, head='mlp', num_labels=num_labels)

print(summary(model, train_dataset[0]['mel_spectogram'].shape))