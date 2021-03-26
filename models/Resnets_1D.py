import torch
import torch.nn as nn
import torch.nn.functional as F

class Res15(nn.Module):
    
    def __init__(self, in_channels, n_maps):
        super().__init__()
        n_maps = n_maps
        self.conv0 = nn.Conv1d(in_channels, n_maps, 3, padding=1, bias=False)
        self.n_layers = n_layers = 13
        dilation = True
        if dilation:
            self.convs = [nn.Conv1d(n_maps, n_maps, 3, padding=int(2 ** (i // 3)), dilation=int(2 ** (i // 3)),
                                    bias=False) for i in range(n_layers)]
        else:
            self.convs = [nn.Conv1d(n_maps, n_maps, 3, padding=1, dilation=1,
                                    bias=False) for _ in range(n_layers)]
        for i, conv in enumerate(self.convs):
            self.add_module("bn{}".format(i + 1), nn.BatchNorm1d(n_maps, affine=False))
            self.add_module("conv{}".format(i + 1), conv)

    def forward(self, audio_signal, length=None):
        x = audio_signal

        # In case the MEL has a 1 channel
        if x.shape[1] == 1 and len(x.shape) > 3:
            x = torch.squeeze(x, dim=1)

        #x = audio_signal.unsqueeze(1)
        for i in range(self.n_layers + 1):
            y = F.relu(getattr(self, "conv{}".format(i))(x))
            if i == 0:
                if hasattr(self, "pool"):
                    y = self.pool(y)
                old_x = y
            if i > 0 and i % 2 == 0:
                x = y + old_x
                old_x = x
            else:
                x = y
            if i > 0:
                x = getattr(self, "bn{}".format(i))(x)
        x = x.view(x.size(0), x.size(1), -1)  # shape: (batch, feats, o3)
        x = torch.mean(x, 2)
        return x
        #return x.unsqueeze(-2), length



def ResNet15(in_channels=40, n_maps=45):
    return {'backbone': Res15(in_channels=in_channels, n_maps=n_maps), 'dim': 45}


def ResNet18(in_channels=1):
    return {'backbone': ResNet(BasicBlock, [2, 2, 2, 2], in_channels=in_channels), 'dim': 512}

def ResNet50(in_channels=1):
    return {'backbone': ResNet(Bottleneck, [3, 4, 6, 3], in_channels=in_channels), 'dim': 2048}

# reference
# https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325
def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def check_params():

    model = ResNet18()
    print('ResNet18: ', get_n_params(model))

    model = ResNet50()
    print('ResNet50: ', get_n_params(model))

# check_params()
