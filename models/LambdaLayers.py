import torch
from torch import nn, einsum
from einops import rearrange
import torch.nn.functional as F

class LambdaLayer1D(nn.Module):
    def __init__(self, dim, in_tokens, out_tokens, heads=4, k=16, u=1, m=23):
        super(LambdaLayer1D, self).__init__()
        
        self.kk, self.uu, self.vv, self.mm, self.heads = k, u, dim // heads, m, heads
        self.local_context = True if m > 0 else False
        self.padding = (m - 1) // 2

        self.in_tokens = in_tokens

        # Input should be: batch, tokens, dim 

        self.queries = nn.Sequential(
            nn.Linear(dim, k*heads, bias=True),
            nn.BatchNorm1d(in_tokens)
            #nn.Conv2d(in_channels, k * heads, kernel_size=1, bias=False),
            #nn.BatchNorm2d(k * heads)
        )
        self.keys = nn.Sequential(
            nn.Linear(dim, k*u, bias=True)
            #nn.Conv2d(in_channels, k * u, kernel_size=1, bias=False),
        )
        self.values = nn.Sequential(
            nn.Linear(dim, self.vv * u, bias=True),
            nn.BatchNorm1d(in_tokens)
            #nn.Conv2d(in_channels, self.vv * u, kernel_size=1, bias=False),
            #nn.BatchNorm2d(self.vv * u)
        )

        self.softmax = nn.Softmax(dim=-1)

        if self.local_context:
            #self.embedding = nn.Parameter(torch.randn([self.kk, self.uu, 1, m, m]), requires_grad=True)
            self.embedding = nn.Parameter(torch.randn([self.kk, self.uu, 1, m]), requires_grad=True)
        else:
            self.embedding = nn.Parameter(torch.randn([self.kk, self.uu]), requires_grad=True)

    def forward(self, x):
        n_batch, tokens, dim = x.size()

        queries = self.queries(x).view(n_batch, self.heads, self.in_tokens, self.kk)
        softmax = self.softmax(self.keys(x).view(n_batch, self.uu, self.in_tokens, self.kk))
        values = self.values(x).view(n_batch, self.uu, self.in_tokens, self.vv)
        #queries = self.queries(x).view(n_batch, self.heads, self.kk, w * h) # b, heads, k // heads, w * h
        #softmax = self.softmax(self.keys(x).view(n_batch, self.kk, self.uu, w * h)) # b, k, uu, w * h
        #values = self.values(x).view(n_batch, self.vv, self.uu, w * h) # b, v, uu, w * h

        print('Shape of Q (Wq x X): {}'.format(queries.shape))
        print('Shape of K (Wk x C): {}'.format(softmax.shape))
        print('Shape of V (Wv x C): {}'.format(values.shape))

        lambda_c = torch.einsum('bumk,bumv->bkv', softmax, values)
        y_c = torch.einsum('bhnk,bkv->bhvn', queries, lambda_c)
        #lambda_c = torch.einsum('bkum,bvum->bkv', k, v)
        #y_c = torch.einsum('bhkn,bkv->bhvn', queries, lambda_c)
        
        print('Shape of context lambda: {}'.format(lambda_c.shape))
        print('Shape of context output: {}'.format(y_c.shape))
        
        if self.local_context:
            
            # Values here are B U N V
            values = values.view(n_batch, self.uu, -1, self.vv)
            #values = values.view(n_batch, self.uu, -1, w, h)
            
            lambda_p = F.conv2d(values, self.embedding, padding=(0, self.padding))
            #lambda_p = F.conv3d(values, self.embedding, padding=(0, self.padding, self.padding))
            #lambda_p = lambda_p.view(n_batch, self.kk, self.vv, w * h)

            y_p = torch.einsum('bhnk,bknv->bhvn', queries, lambda_p)
            #y_p = torch.einsum('bhkn,bkvn->bhvn', queries, lambda_p)

            print('Shape of positional lambda: {}'.format(lambda_p.shape))
            print('Shape of positional output: {}'.format(y_p.shape))
        
        else:
            lambda_p = torch.einsum('ku,bvun->bkvn', self.embedding, values)
            y_p = torch.einsum('bhkn,bkvn->bhvn', queries, lambda_p)

        out = y_c + y_p
        print('Output shape: {}'.format(out.shape))

        # Output reshaping operation: bn(hv) --> bnd
        out = out.contiguous().view(n_batch, -1, dim)

        return out


class LambdaLayer1D_new(nn.Module):
    def __init__(self, dim, in_tokens, out_tokens, heads=4, k=16, u=1, m=23):
        super(LambdaLayer1D_new, self).__init__()
        
        self.kk, self.uu, self.vv, self.mm, self.heads = k, u, out_tokens // heads, m, heads
        self.local_context = True if m > 0 else False
        self.padding = (m - 1) // 2

        self.in_tokens = in_tokens
        self.out_tokens = out_tokens
        self.dim = dim

        # Input should be: batch, tokens, dim 

        self.queries = nn.Sequential(
            nn.Linear(in_tokens, k*heads, bias=True),
            nn.BatchNorm1d(dim)
            #nn.Conv2d(in_channels, k * heads, kernel_size=1, bias=False),
            #nn.BatchNorm2d(k * heads)
        )
        self.keys = nn.Sequential(
            nn.Linear(in_tokens, k*u, bias=True)
            #nn.Conv2d(in_channels, k * u, kernel_size=1, bias=False),
        )
        self.values = nn.Sequential(
            nn.Linear(in_tokens, self.vv * u, bias=True),
            nn.BatchNorm1d(dim)
            #nn.Conv2d(in_channels, self.vv * u, kernel_size=1, bias=False),
            #nn.BatchNorm2d(self.vv * u)
        )

        self.softmax = nn.Softmax(dim=-1)

        if self.local_context:
            #self.embedding = nn.Parameter(torch.randn([self.kk, self.uu, 1, m, m]), requires_grad=True)
            self.embedding = nn.Parameter(torch.randn([self.kk, self.uu, 1, m]), requires_grad=True)
        else:
            self.embedding = nn.Parameter(torch.randn([self.kk, self.uu]), requires_grad=True)

    def forward(self, x):
        n_batch, _, _ = x.size()

        print(x.shape)
        x = x.permute(0,2,1)
        print(x.shape)
        queries = self.queries(x).view(n_batch, self.heads, self.kk, self.dim)
        softmax = self.softmax(self.keys(x).view(n_batch, self.kk, self.uu, self.dim))
        values = self.values(x).view(n_batch, self.vv, self.uu, self.dim)
        #queries = self.queries(x).view(n_batch, self.heads, self.kk, w * h) # b, heads, k // heads, w * h
        #softmax = self.softmax(self.keys(x).view(n_batch, self.kk, self.uu, w * h)) # b, k, uu, w * h
        #values = self.values(x).view(n_batch, self.vv, self.uu, w * h) # b, v, uu, w * h

        print('Shape of Q (Wq x X): {}'.format(queries.shape))
        print('Shape of K (Wk x C): {}'.format(softmax.shape))
        print('Shape of V (Wv x C): {}'.format(values.shape))

        lambda_c = torch.einsum('bkum,bvum->bkv', softmax, values)
        y_c = torch.einsum('bhkn,bkv->bhvn', queries, lambda_c)
        
        print('Shape of context lambda: {}'.format(lambda_c.shape))
        print('Shape of context output: {}'.format(y_c.shape))
        
        if self.local_context:
            
            # Values here are B U N V
            values = values.view(n_batch, self.uu, -1, self.dim)
            #values = values.view(n_batch, self.uu, -1, w, h)
            
            lambda_p = F.conv2d(values, self.embedding, padding=(0, self.padding))
            #lambda_p = F.conv3d(values, self.embedding, padding=(0, self.padding, self.padding))
            #lambda_p = lambda_p.view(n_batch, self.kk, self.vv, w * h)

            y_p = torch.einsum('bhkn,bkvn->bhvn', queries, lambda_p)

            print('Shape of positional lambda: {}'.format(lambda_p.shape))
            print('Shape of positional output: {}'.format(y_p.shape))
        
        else:
            lambda_p = torch.einsum('ku,bvun->bkvn', self.embedding, values)
            y_p = torch.einsum('bhkn,bkvn->bhvn', queries, lambda_p)

        out = y_c + y_p
        print('Output shape: {}'.format(out.shape))

        # Output reshaping operation: bn(hv) --> bnd
        out = out.contiguous().view(n_batch, -1, self.dim)

        return out


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