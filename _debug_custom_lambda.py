import torch
from models.LambdaLayers import LambdaLayer1D
from models.LambdaLayers import LambdaLayer2D

def main():
    # Batch size, number of inputs, dimension of the input
    audio_input_1d = torch.randn(200, 8, 300)
    debug_1d(audio_input_1d)

    img_input_2d = torch.randn(1, 32, 64, 64)
    debug_2d(img_input_2d)


def debug_1d(tensor):
    batch, d_in, dim = tensor.shape
    d_out = d_in
    
    print('1D Lambda layer (Dense)')
    layer = LambdaLayer1D(d_in, d_out, heads=4, k=16, u=1, m=23, layer_type='dense', dim=dim)
    print('Input: {}'.format(tensor.shape))
    print('Output: {}'.format(layer(tensor).shape))

    print('1D Lambda layer (Conv)')
    layer = LambdaLayer1D(d_in, d_out, heads=4, k=16, u=1, m=23, layer_type='conv')
    print('Input: {}'.format(tensor.shape))
    print('Output: {}'.format(layer(tensor).shape))


def debug_2d(tensor):
    batch, in_channels, h, w = tensor.shape
    out_channels = in_channels
    
    print('2D Lambda layer (Conv)')
    layer = LambdaLayer2D(in_channels, out_channels, heads=4, k=16, u=1, m=23)

    print('Input: {}'.format(tensor.shape))
    print('Output: {}'.format(layer(tensor).shape))


if __name__ == "__main__":
    main()