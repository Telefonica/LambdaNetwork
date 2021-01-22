from utils.common_config import get_dataset, get_train_dataloader, get_val_transformations
from utils.config import create_config
from utils.audioutils import save_wav, open_wavfile
import numpy as np
import torch

import matplotlib.pyplot as plt
from torchaudio import load

config_exp = './configs/lambdaResnet2D_supervised.yml'
config_env = './configs/env.yml'

p = create_config(config_env, config_exp)

transforms = get_val_transformations(p)
dataset = get_dataset(p, transform=transforms, to_augmented_dataset=False, subset="validation")
dataloader = get_train_dataloader(p, dataset)

audio = dataset[0]['audio']
print(dataset[0]['label'])

background = load('datasets/SpeechCommands/speech_commands_v0.02/_background_noise_/doing_the_dishes.wav')
back_noise_tensor = background[0][0, 0:16000]

# Add a random value summed up
combined_audio = audio + 0.1*back_noise_tensor

audio_int = (audio.numpy()[0]*32768.0).astype(np.int16)
back_noise = (background[0].numpy()[0]*32768.0).astype(np.int16)
combined_audio_int = (combined_audio.numpy()*32768.0).astype(np.int16)

save_wav(back_noise, 'back.wav')
save_wav(audio_int, 'tensor.wav')
save_wav(combined_audio_int, 'combined.wav')

plt.figure()
plt.plot(audio_int)
plt.show()
