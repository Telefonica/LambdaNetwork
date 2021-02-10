from utils.common_config import get_dataset, get_train_dataloader, get_train_transformations, get_val_dataloader
from utils.config import create_config
from utils.audioutils import save_wav, open_wavfile
import numpy as np

config_exp = './configs/google_commands/LambdaResnet2D.yml'
config_env = './configs/env.yml'

p = create_config(config_env, config_exp)

transforms = get_train_transformations(p)
dataset = get_dataset(p, transform=transforms, subset="train")
dataloader = get_val_dataloader(p, dataset)

for i, batch in enumerate(dataloader):
    print('MEL SPECTOGRAM: {}'.format(batch['mel_spectogram'].shape))
    print('AUDIO: {}'.format(batch['audio'].shape))

    print('LABELS: {}'.format(batch['label']))
    print('LABELS: {}'.format(batch['target']))

    import torch
    print(torch.nn.functional.one_hot(batch['target']))

    audio_tensor = batch['audio']
    audio_wav = (audio_tensor.numpy()[0]*32768.0).astype(np.int16)
    save_wav(audio_wav, 'augmented_audio_{}.wav'.format(batch['label'][0]))

    break