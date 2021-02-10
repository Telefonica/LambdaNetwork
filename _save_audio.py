from utils.common_config import get_dataset, get_train_dataloader, get_val_transformations
from utils.config import create_config
from utils.audioutils import save_wav, open_wavfile
import numpy as np
import torch

import matplotlib.pyplot as plt
from torchaudio import load

config_exp = './configs/google_commands/LambdaResnet2D.yml'
config_env = './configs/env.yml'

p = create_config(config_env, config_exp)

transforms = get_val_transformations(p)
dataset = get_dataset(p, transform=transforms, subset="validation")
dataloader = get_train_dataloader(p, dataset)

audio = dataset[0]['audio']
print(dataset[0]['label'])

from data.augment import AddBackgroundNoiseSNR

transform = AddBackgroundNoiseSNR(p=1.0, SNR_range_db=(0, 15))
combined_audio = transform(audio)

audio_int = (audio.numpy()[0]*32768.0).astype(np.int16)
combined_audio_int = (combined_audio.numpy()[0]*32768.0).astype(np.int16)

#save_wav(back_noise, 'back.wav')
save_wav(audio_int, 'tensor.wav')
save_wav(combined_audio_int, 'combined.wav')

print(audio_int.shape)
print(combined_audio_int.shape)

plt.figure()
plt.plot(audio_int)
plt.plot(combined_audio_int)
plt.show()
