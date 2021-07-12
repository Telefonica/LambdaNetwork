import torch
from torch import nn, einsum
from einops import rearrange
import torch.nn.functional as F


class LambdaLayer2D(nn.Module):
    def __init__(self, in_channels, out_channels, heads=4, k=16, u=1, m=23):
        super(LambdaLayer2D, self).__init__()
        self.kk, self.uu, self.vv, self.mm, self.heads = k, u, out_channels // heads, m, heads
        self.local_context = True if m > 0 else False
        self.padding = (m - 1) // 2

        self.queries = nn.Sequential(
            nn.Conv2d(in_channels, k * heads, kernel_size=1, bias=False),
            nn.BatchNorm2d(k * heads)
        )
        self.keys = nn.Sequential(
            nn.Conv2d(in_channels, k * u, kernel_size=1, bias=False),
        )
        self.values = nn.Sequential(
            nn.Conv2d(in_channels, self.vv * u, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.vv * u)
        )

        self.softmax = nn.Softmax(dim=-1)

        if self.local_context:
            self.embedding = nn.Parameter(torch.randn([self.kk, self.uu, 1, m, m]), requires_grad=True)
        else:
            self.embedding = nn.Parameter(torch.randn([self.kk, self.uu]), requires_grad=True)

    def forward(self, x):
        n_batch, C, w, h = x.size()
        
        queries = self.queries(x).view(n_batch, self.heads, self.kk, w * h) # b, heads, k // heads, w * h
        softmax = self.softmax(self.keys(x).view(n_batch, self.kk, self.uu, w * h)) # b, k, uu, w * h
        values = self.values(x).view(n_batch, self.vv, self.uu, w * h) # b, v, uu, w * h

        lambda_c = torch.einsum('bkum,bvum->bkv', softmax, values)
        y_c = torch.einsum('bhkn,bkv->bhvn', queries, lambda_c)

        if self.local_context:
            values = values.view(n_batch, self.uu, -1, w, h)
            lambda_p = F.conv3d(values, self.embedding, padding=(0, self.padding, self.padding))
            lambda_p = lambda_p.view(n_batch, self.kk, self.vv, w * h)
            y_p = torch.einsum('bhkn,bkvn->bhvn', queries, lambda_p)
        else:
            lambda_p = torch.einsum('ku,bvun->bkvn', self.embedding, values)
            y_p = torch.einsum('bhkn,bkvn->bhvn', queries, lambda_p)

        out = y_c + y_p
        out = out.contiguous().view(n_batch, -1, w, h)

        return out


class LambdaLayer1D(nn.Module):
    def __init__(self, d_in, d_out, heads=4, k=16, u=1, m=23, layer_type='conv', dim=None):
        super(LambdaLayer1D, self).__init__()
        self.kk, self.uu, self.vv, self.mm, self.heads = k, u, d_out // heads, m, heads
        self.local_context = True if m > 0 else False
        self.padding = (m - 1) // 2
        self.layer_type = layer_type

        if layer_type == 'conv':
            self.queries = nn.Sequential(
                nn.Conv1d(d_in, k * heads, kernel_size=1, bias=False),
                nn.BatchNorm1d(k * heads)
            )
            self.keys = nn.Sequential(
                nn.Conv1d(d_in, k * u, kernel_size=1, bias=False),
            )
            self.values = nn.Sequential(
                nn.Conv1d(d_in, self.vv * u, kernel_size=1, bias=False),
                nn.BatchNorm1d(self.vv * u)
            )
        if layer_type == 'dense':
            self.queries = nn.Sequential(
                nn.Linear(d_in, k*heads, bias=True),
                nn.BatchNorm1d(dim)
            )
            self.keys = nn.Sequential(
                nn.Linear(d_in, k * u, bias=True),
            )
            self.values = nn.Sequential(
                nn.Linear(d_in, self.vv * u, bias=True),
                nn.BatchNorm1d(dim)
            )

        self.softmax = nn.Softmax(dim=-1)

        if self.local_context:
            self.embedding = nn.Parameter(torch.randn([self.kk, self.uu, 1, m]), requires_grad=True)
        else:
            self.embedding = nn.Parameter(torch.randn([self.kk, self.uu]), requires_grad=True)

    def forward(self, x):
        n_batch, d, m = x.size()

        # Permute the variable to compute the q,k,v with a dense operator
        if self.layer_type == 'dense':
            x = x.permute(0,2,1)
        
        queries = self.queries(x).view(n_batch, self.heads, self.kk, m) # b, heads, k // heads, m
        softmax = self.softmax(self.keys(x).view(n_batch, self.kk, self.uu, m)) # b, k, u, m
        values = self.values(x).view(n_batch, self.vv, self.uu, m) # b, v, u, m

        lambda_c = torch.einsum('bkum,bvum->bkv', softmax, values)
        y_c = torch.einsum('bhkn,bkv->bhvn', queries, lambda_c)

        if self.local_context:
            values = values.view(n_batch, self.uu, -1, m)
            lambda_p = F.conv2d(values, self.embedding, padding=(0, self.padding))
            lambda_p = lambda_p.view(n_batch, self.kk, self.vv, m)
            y_p = torch.einsum('bhkn,bkvn->bhvn', queries, lambda_p)
        else:
            lambda_p = torch.einsum('ku,bvun->bkvn', self.embedding, values)
            y_p = torch.einsum('bhkn,bkvn->bhvn', queries, lambda_p)

        out = y_c + y_p
        out = out.contiguous().view(n_batch, -1, m)

        return out