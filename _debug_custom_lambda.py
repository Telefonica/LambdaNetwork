import torch
from models.LambdaLayers import LambdaLayer1D_new
from models.LambdaLayers import LambdaLayer1D
from models.LambdaLayers import LambdaLayer2D


def main():
    # Batch size, number of inputs, dimension of the input
    #audio_input_1d = torch.randn(200, 5, 300)
    audio_input_1d = torch.randn(200, 8, 300)
    debug_1d(audio_input_1d)

    #img_input_2d = torch.randn(1, 32, 64, 64)
    #debug_2d(img_input_2d)


def debug_1d(tensor):

    batch, in_tokens, dim = tensor.shape
    out_tokens = in_tokens
    layer = LambdaLayer1D_new(dim, in_tokens, out_tokens, heads=4, k=16, u=1, m=23)

    print(layer(tensor).shape)

def debug_2d(tensor):
    batch, channels, h, w = tensor.shape
    out_channels = in_channels
    layer = LambdaLayer2D(in_channels, out_channels, heads=4, k=16, u=1, m=23)

    print(layer(tensor).shape)


if __name__ == "__main__":
    main()