import os
import math
import numpy as np
import torch
import torch.nn as nn
import data.augment as augment

# On this case, p is config file and p['criterion_kwargs'] includes the temperature
def get_criterion(p):
    if p['criterion'] == 'supervised':
        from losses.losses import CrossEntropyLoss
        criterion = CrossEntropyLoss()
    
    return criterion

def get_model(p):

    if p['backbone'] == 'LambdaResnet':
        if p['setup'] == '2D':
            from models import LambdaResnets_2D as LambdaResnets
            backbone = LambdaResnets.LambdaResNet15(in_channels=1, n_maps=44)
        if p['setup'] == '1D':
            #from models import LambdaResnets_1D as LambdaResnets
            from models import new_lambda as LambdaResnets
            backbone = LambdaResnets.LambdaResNet15(in_channels=p['spectogram_kwargs']['n_mels'], k=2)
    
    elif p['backbone'] == 'Resnet':
        if p['setup'] == '2D':
            from models import Resnets_2D as Resnets
            backbone = Resnets.ResNet15(in_channels=1, n_maps=45)
        if p['setup'] == '1D':
            #from models import Resnets_1D as Resnets
            from models import TC_Resnet as Resnets
            backbone = Resnets.TCResnet14(in_channels=p['spectogram_kwargs']['n_mels'], n_maps=45)

    from models.heads import SupervisedModel
    model = SupervisedModel(backbone, **p['model_kwargs'])
    
    return model


def get_dataset(p, transform, subset=None):
    
    if p['db_name'] == 'google_commands':
        from data.datasets import SpeechCommands
        dataset = SpeechCommands(num_labels=p['num_labels'], subset=subset)

    from data.custom_dataset import AudioDataset
    dataset = AudioDataset(dataset, transform=transform)

    if p['frontend'] == 'mel':
        from data.custom_dataset import MelDataset
        dataset = MelDataset(dataset, **p['spectogram_kwargs'])
    
    return dataset


def get_train_dataloader(p, dataset):
    
    '''# Get weighted sampler if we use less than all the commands
    if p['db_name'] == 'google_commands' and p['num_labels'] != 35:
        from data.samplers import get_SpeechCommandsSampler
        sampler = get_SpeechCommandsSampler(p, dataset)

        return torch.utils.data.DataLoader(dataset, num_workers=p['num_workers'], 
            batch_size=p['batch_size'], sampler=sampler, pin_memory=True)
    '''
    return torch.utils.data.DataLoader(dataset, num_workers=p['num_workers'], 
            batch_size=p['batch_size'], pin_memory=True,
            drop_last=True, shuffle=True)
    

def get_val_dataloader(p, dataset):
    return torch.utils.data.DataLoader(dataset, num_workers=p['num_workers'],
            batch_size=p['batch_size'], pin_memory=True,
            drop_last=False, shuffle=False)


def get_train_transformations(p):
    if p['transformation_strategy'] == 'audio':
        return nn.Sequential(
            #augment.PadTrim(p=0.5),                                                        # NOT WORK: DONT KNOW ?
            augment.VolTransform(p=p['transformation_kwargs']['volume']['p']),                # Increases / Decreases volume
            #augment.VadTransform(p=0.5),                                                   # NOT NEEDED: Tries to trim Silence in the audio
            #augment.FadeTransform(p=0.5),                                                  # NOT WORK: FIX RANDOM RANGES
            augment.CropTransform(p=p['transformation_kwargs']['crop']['p']),               # Randomply crop a small part of the signal
            #augment.RIRTransform(p=0.5),                                                   # Room impulse response
            #augment.GaussianSNRTransform(p=0.5),                                           # Adding Gaussian noise
            augment.TimeStretchTransform(p=p['transformation_kwargs']['time_strech']['p']), # Increase / Decrease time without mod its pitch
            augment.PitchShiftTransform(p=p['transformation_kwargs']['pitch_shift']['p']),  # Shif the pitch of the signal
            augment.ShiftTransform(p=p['transformation_kwargs']['temporal_shift']['p']),    # Shift the audio signal temporally
            augment.ClippingDistortionTransform(p=p['transformation_kwargs']['clipping_distortion']['p']), # Saturation distortion the audio signal
            #augment.AddBackgroundNoise(p=0.7),                                             # Add background noise randomly picked (volume)
            augment.AddBackgroundNoiseSNR(p=p['transformation_kwargs']['background_noise']['p']), # Add background noise randomly picked (SNR)
            augment.LengthTransform()                                                       # After all the transforms, keep the same length
        )

def get_val_transformations(p):
    if p['transformation_strategy'] == 'audio':
        return nn.Sequential(
            augment.LengthTransform()
        )

def get_optimizer(p, model, cluster_head_only=False):
    if cluster_head_only: # Only weights in the cluster head will be updated 
        for name, param in model.named_parameters():
                if 'cluster_head' in name:
                    param.requires_grad = True 
                else:
                    param.requires_grad = False 
        params = list(filter(lambda p: p.requires_grad, model.parameters()))
        assert(len(params) == 2 * p['num_heads'])

    else:
        params = model.parameters()
                
    if p['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(params, **p['optimizer_kwargs'])

    elif p['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(params, **p['optimizer_kwargs'])
    
    else:
        raise ValueError('Invalid optimizer {}'.format(p['optimizer']))

    return optimizer


def adjust_learning_rate(p, optimizer, epoch):
    lr = p['optimizer_kwargs']['lr']
    
    if p['scheduler'] == 'cosine':
        eta_min = lr * (p['scheduler_kwargs']['lr_decay_rate'] ** 3)
        lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / p['epochs'])) / 2
         
    elif p['scheduler'] == 'step':
        steps = np.sum(epoch > np.array(p['scheduler_kwargs']['lr_decay_epochs']))
        if steps > 0:
            lr = lr * (p['scheduler_kwargs']['lr_decay_rate'] ** steps)

    elif p['scheduler'] == 'constant':
        lr = lr

    else:
        raise ValueError('Invalid learning rate schedule {}'.format(p['scheduler']))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

'''
def get_model_sincnet(p):

    if p['setup'] == '2D':
        from models import LambdaResnets_2D as LambdaResnets
        if p['frontend'] == 'mel':
            #backbone = LambdaResnets.LambdaResNet18(in_channels=1)
            backbone = LambdaResnets.LambdaResNet15_2d(in_channels=1)
        
        if p['frontend'] == 'sincnet':
            backbone = LambdaResnets.LambdaResNet18(in_channels=1)

            # BIG TODO HERE: Move this variables to the config file
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

            from models.SincNet import SincNet
            sincnet = SincNet(CNN_arch)
            # END TODO
    
    if p['setup'] == '1D':
        from models import LambdaResnets_1D as LambdaResnets

        if p['frontend'] == 'mel':
            #backbone = LambdaResnets.LambdaResNet18(in_channels=p['spectogram_kwargs']['n_mels'])
            backbone = LambdaResnets.LambdaResNet15_1d(in_channels=p['spectogram_kwargs']['n_mels'])
    
        elif p['frontend'] == 'sincnet':

            # BIG TODO HERE: Move this variables to the config file
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

            from models.SincNet import SincNet
            sincnet = SincNet(CNN_arch)
            # END TODO

            backbone = LambdaResnets.LambdaResNet18(in_channels=60)
        
        elif p['frontend'] == 'raw':
            backbone = LambdaResnets.LambdaResNet18(in_channels=1)

    # TODO FIX FOR RESNETS
    if p['backbone'] == 'Resnet50':
        from models import Resnets_2D
        backbone = Resnets_2D.ResNet50(in_channels=1)
    
    if p['backbone'] == 'Resnet18':
        from models import Resnets_2D
        backbone = Resnets_2D.ResNet18(in_channels=1)\
    
    if p['backbone'] == 'Resnet15':
        from models import Resnets_2D
        backbone = Resnets_2D.ResNet15(in_channels=1)

    if p['backbone'] == 'Resnet15_1d':
        from models import Resnets_1D
        backbone = Resnets_1D.ResNet15(in_channels=p['spectogram_kwargs']['n_mels'])

    if p['backbone'] == 'LightResnet15':
        from models import LightResnets_2D
        backbone = LightResnets_2D.ResNet15(in_channels=1)


    from models.heads import SupervisedModel

    # Add the frontend network
    if p['frontend'] == 'sincnet':
        model = SupervisedModel(backbone, frontend=sincnet, **p['model_kwargs'])
    else:
        model = SupervisedModel(backbone, **p['model_kwargs'])
    
    return model
'''