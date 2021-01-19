from utils.common_config import get_dataset, get_train_dataloader, split_dataset, get_train_transformations, get_eval_transformations
from utils.config import create_config
import torch

import librosa.display
import matplotlib.pyplot as plt

config_exp = './configs/lambdaResnet2D_supervised.yml'
config_env = './configs/env.yml'

print('Dataset loaded')
p = create_config(config_env, config_exp)
transformations = get_train_transformations(p)
#transformations = get_eval_transformations(p)
dataset = get_dataset(p, transform=transformations, to_augmented_dataset=True)
dataloader = get_train_dataloader(p, dataset)

print('BATCH DIMENSIONS:')
for i, batch in enumerate(dataloader):
    print(batch['audio'].shape)
    print(batch['mel_spectogram'].shape)
    break