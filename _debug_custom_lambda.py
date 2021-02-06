import torch
from models.LambdaLayers import LambdaLayer1D
from models.LambdaLayers import LambdaLayer2D
from torchscan import summary
from utils.common_config import get_val_transformations
from utils.config import create_config
import torchaudio

def main():

    config_exp = './configs/lambdaResnet2D_supervised.yml'
    config_env = './configs/env.yml'

    p = create_config(config_env, config_exp)
    mel_transform = get_mel_transformation(**p['spectogram_kwargs'])

    # Batch size, number of inputs, dimension of the input (sec x sampling_rate)
    audio_input_1d = torch.randn(1, 45, 1*16000)
    debug_1d(audio_input_1d)

    # Batch size, number of inputs, dimension of the input (w x h)
    audio_input_2d = torch.log(mel_transform(audio_input_1d)+0.001)
    debug_2d(audio_input_2d)


def debug_1d(tensor):
    batch, d_in, dim = tensor.shape
    d_out = d_in
    
    print('1D Lambda layer (Dense)')
    layer = LambdaLayer1D(d_in, d_out, heads=4, k=16, u=1, m=23, layer_type='dense', dim=dim)

    print('FLOPS for lambda layer 1D (dense):')
    print(summary(layer, tensor[0].shape))
    print('Input: {}'.format(tensor.shape))
    print('Output: {}'.format(layer(tensor).shape))

    print('1D Lambda layer (Conv)')
    layer = LambdaLayer1D(d_in, d_out, heads=4, k=16, u=1, m=23, layer_type='conv')

    print('FLOPS for lambda layer 1D (conv):')
    print(summary(layer, tensor[0].shape))
    print('Input: {}'.format(tensor.shape))
    print('Output: {}'.format(layer(tensor).shape))


def debug_2d(tensor):
    batch, in_channels, h, w = tensor.shape
    out_channels = in_channels
    
    print('2D Lambda layer (Conv)')
    layer = LambdaLayer2D(in_channels, out_channels, heads=4, k=16, u=1, m=23)

    print('FLOPS for lambda layer 2D:')
    print(summary(layer, tensor[0].shape))
    print('Input: {}'.format(tensor.shape))
    print('Output: {}'.format(layer(tensor).shape))


def get_mel_transformation(sample_rate, n_fft, n_mels, win_size, win_stride):
    return torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            win_length=int(win_size*sample_rate),
            hop_length=int(win_stride*sample_rate) )


if __name__ == "__main__":
    main()