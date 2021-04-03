import os
import math
import numpy as np
import torch
import torch.nn as nn
import data.augment as augment
from utils.config import mkdir_if_missing


def get_criterion(p):
    if p['criterion'] == 'supervised':
        from losses.losses import CrossEntropyLoss
        criterion = CrossEntropyLoss()
    
    return criterion


def get_model(p):
    if p['backbone'] == 'LambdaResnet':
        from models.LambdaResnet import LambdaResNet18
        backbone = LambdaResNet18(in_channels=p['spectogram_kwargs']['n_mels'], k=p['channel_k'])
    
    elif p['backbone'] == 'TCResnet':
        from models.TCResnet import TCResnet14
        backbone = TCResnet14(in_channels=p['spectogram_kwargs']['n_mels'], k=p['channel_k'])
    
    elif p['backbone'] == 'Resnet':
        from models.Resnet import ResNet15
        backbone = ResNet15(in_channels=1, n_maps=45)

    from models.heads import ClassificationModel
    model = ClassificationModel(backbone, **p['model_kwargs'])
    
    return model


def get_dataset(p, transform, subset=None):
    mkdir_if_missing('./datasets/')
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
    
    '''
    Not giving better results  yet
    
    # Get weighted sampler if we use less than all the commands
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
            augment.VolTransform(p=p['transformation_kwargs']['volume']['p']),                # Increases / Decreases volume
            augment.CropTransform(p=p['transformation_kwargs']['crop']['p']),               # Randomply crop a small part of the signal
            augment.TimeStretchTransform(p=p['transformation_kwargs']['time_strech']['p']), # Increase / Decrease time without mod its pitch
            augment.PitchShiftTransform(p=p['transformation_kwargs']['pitch_shift']['p']),  # Shif the pitch of the signal
            augment.ShiftTransform(p=p['transformation_kwargs']['temporal_shift']['p']),    # Shift the audio signal temporally
            augment.ClippingDistortionTransform(p=p['transformation_kwargs']['clipping_distortion']['p']), # Saturation distortion the audio signal
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