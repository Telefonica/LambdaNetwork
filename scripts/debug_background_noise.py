from utils.common_config import get_dataset, get_train_dataloader, get_val_transformations
from utils.config import create_config
import numpy as np
import torch

import matplotlib.pyplot as plt
from torchaudio import load

config_exp = './configs/google_commands/LambdaResnet18_35.yml'
config_env = './configs/env.yml'

# Get the config file and get the raw waveform
p = create_config(config_env, config_exp)
p['frontend'] = 'raw'

transforms = get_val_transformations(p)
dataset = get_dataset(p, transform=transforms, subset="validation")

audio = dataset[0]['input']
print('Audio: {}'.format(dataset[0]['label']))

from data.augment import AddBackgroundNoiseSNR
transform = AddBackgroundNoiseSNR(p=1.0, SNR_range_db=(0, 15))
combined_audio = transform(audio)

audio_int = (audio.numpy()[0]*32768.0).astype(np.int16)
combined_audio_int = (combined_audio.numpy()[0]*32768.0).astype(np.int16)

#save_wav(back_noise, 'background_noise.wav')
#save_wav(audio_int, 'original_audio.wav')
#save_wav(combined_audio_int, 'transformed_audio.wav')

plt.figure()
plt.plot(combined_audio_int, label='Audio + Background noise')
plt.plot(audio_int, label='Original Audio')
plt.legend()
plt.show()