import torch
from models.SincNet import SincNet

# [cnn]
fs = 16000
cw_len = 1000
wlen = int(fs*cw_len/1000.00)
cnn_N_filt = 80, 60, 60
cnn_len_filt = 251, 5, 5
cnn_max_pool_len=3, 3, 3
cnn_use_laynorm_inp=True
cnn_use_batchnorm_inp=False
cnn_use_laynorm=True, True, True
cnn_use_batchnorm=False, False, False
cnn_act= ['relu','relu','relu']
cnn_drop=0.0, 0.0, 0.0

# [dnn]
fc_lay=2048,2048,2048
fc_drop=[0.0,0.0,0.0]
fc_use_laynorm_inp=True
fc_use_batchnorm_inp=False
fc_use_batchnorm=True,True,True
fc_use_laynorm=False,False,False
fc_act=['leaky_relu','leaky_relu','leaky_relu']

# [class]
class_lay=[10]
class_drop=[0.0]
class_use_laynorm_inp=False
class_use_batchnorm_inp=False
class_use_batchnorm=[False]
class_use_laynorm=[False]
class_act=['softmax']

CNN_arch = {'input_dim': wlen,
    'fs': fs,
    'cnn_N_filt': cnn_N_filt,
    'cnn_len_filt': cnn_len_filt,
    'cnn_max_pool_len':cnn_max_pool_len,
    'cnn_use_laynorm_inp': cnn_use_laynorm_inp,
    'cnn_use_batchnorm_inp': cnn_use_batchnorm_inp,
    'cnn_use_laynorm':cnn_use_laynorm,
    'cnn_use_batchnorm':cnn_use_batchnorm,
    'cnn_act': cnn_act,
    'cnn_drop':cnn_drop,          
    }

model = SincNet(CNN_arch)

input = torch.randn(5, 1, 16000)
print(model(input).shape)