from utils.common_config import get_dataset, get_train_dataloader, split_dataset
from utils.config import create_config

config_exp = './configs/lambdaResnet_supervised.yml'
config_env = './configs/env.yml'

p = create_config(config_env, config_exp)

dataset = get_dataset(p, transform='placeholder', to_augmented_dataset=False)
train_dataset, val_dataset = split_dataset(p, dataset, split=0.7)
dataloader = get_train_dataloader(p, dataset)

for i, batch in enumerate(dataloader):
    print('MEL SPECTOGRAM: {}'.format(batch['mel_spectogram'].shape))
    print('AUDIO: {}'.format(batch['audio'].shape))

    print('LABELS: {}'.format(batch['label']))
    print('LABELS: {}'.format(batch['categorical_label']))

    import torch
    print(torch.nn.functional.one_hot(batch['categorical_label']))



    
    break

#print(dataset[0])