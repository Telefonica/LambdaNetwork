import torch
from models.LambdaLayers import LambdaLayer1D
from models.LambdaLayers import LambdaLayer2D
from torchscan import summary

def main():
    # Batch size, number of inputs, dimension of the input (sec x sampling_rate)
    audio_input = torch.randn(1, 40, 100)
    debug_1d(audio_input)
    #debug_2d(audio_input)


def debug_1d(tensor):
    channel, d_in, dim = tensor.shape
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
    in_channels, h, w = tensor.shape
    out_channels = in_channels
    
    print('2D Lambda layer (Conv)')
    layer = LambdaLayer2D(in_channels, out_channels, heads=4, k=16, u=1, m=23)

    print('FLOPS for lambda layer 2D:')
    print(summary(layer, tensor.shape))
    print('Input: {}'.format(tensor.shape))
    print('Output: {}'.format(layer(tensor).shape))


if __name__ == "__main__":
    main()