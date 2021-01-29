import os
import math
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as audio_transforms
import data.augment as augment

# On this case, p is config file and p['criterion_kwargs'] includes the temperature
def get_criterion(p):
    if p['criterion'] == 'simclr':
        from losses.losses import SimCLRLoss
        criterion = SimCLRLoss(**p['criterion_kwargs'])

    if p['criterion'] == 'supervised':
        from losses.losses import CrossEntropyLoss
        criterion = CrossEntropyLoss()
    
    return criterion


def get_feature_dimensions_backbone(p):
    if p['backbone'] == 'lambdaResnet18':
        return 512

    elif p['backbone'] == 'lambdaResnet50':
        return 2048

    else:
        raise NotImplementedError


def get_model(p, pretrain_path=None):
    if p['backbone'] == 'lambdaResnet50':
        from models import LambdaResnets
        backbone = LambdaResnets.LambdaResNet50(in_channels=1)
    
    if p['backbone'] == 'lambdaResnet18':
        from models import LambdaResnets
        backbone = LambdaResnets.LambdaResNet18(in_channels=1)

    if p['backbone'] == 'Resnet50':
        from models import Resnets
        backbone = Resnets.ResNet50(in_channels=1)
    
    if p['backbone'] == 'Resnet18':
        from models import Resnets
        backbone = Resnets.ResNet18(in_channels=1)

    if p['setup'] == 'contrastive':
        from models.heads import ContrastiveModel
        model = ContrastiveModel(backbone)

    if p['setup'] == 'supervised':
        from models.heads import SupervisedModel
        model = SupervisedModel(backbone, **p['model_kwargs'])
    
    return model


def get_dataset(p, transform, to_augmented_dataset=False, to_neighbors_dataset=False, split=None, subset=None):
    if p['train_db_name'] == 'google_commands':
        from data.datasets import SpeechCommands
        dataset = SpeechCommands(num_labels=p['num_labels'], subset=subset)

    from data.custom_dataset import MelDataset
    dataset = MelDataset(dataset, **p['spectogram_kwargs'], transform=transform)

    # Wrap into other dataset (__getitem__ changes)
    if to_augmented_dataset: # Dataset returns pairs of augmented audio
        from data.custom_dataset import AugmentedDataset
        dataset = AugmentedDataset(dataset, transform=transform)

    return dataset

def get_train_dataloader(p, dataset):
    return torch.utils.data.DataLoader(dataset, num_workers=p['num_workers'], 
            batch_size=p['batch_size'], pin_memory=True,
            drop_last=True, shuffle=True)


def get_val_dataloader(p, dataset):
    return torch.utils.data.DataLoader(dataset, num_workers=p['num_workers'],
            batch_size=p['batch_size'], pin_memory=True,
            drop_last=False, shuffle=False)


def get_train_transformations(p):
    if p['augmentation_strategy'] == 'audio':
        return nn.Sequential(
            #augment.PadTrim(p=0.5),                    # NOT WORK: DONT KNOW ?
            augment.VolTransform(p=0.6),                # Increases / Decreases volume
            #augment.VadTransform(p=0.5),               # NOT NEEDED: Tries to trim Silence in the audio
            #augment.FadeTransform(p=0.5),              # NOT WORK: FIX RANDOM RANGES
            augment.CropTransform(p=0.2),               # Randomply crop a small part of the signal
            #augment.RIRTransform(p=0.5),               # Room impulse response
            #augment.GaussianSNRTransform(p=0.5),       # Adding Gaussian noise
            augment.TimeStretchTransform(p=0.3),        # Increase / Decrease time without mod its pitch
            augment.PitchShiftTransform(p=0.2),         # Shif the pitch of the signal
            augment.ShiftTransform(p=0.3),              # Shift the audio signal temporally
            augment.ClippingDistortionTransform(p=0.2), # Saturation distortion the audio signal
            #augment.AddBackgroundNoise(p=0.7),         # Add background noise randomly picked (volume)
            augment.AddBackgroundNoiseSNR(p=0.7),       # Add background noise randomly picked (SNR)
            augment.LengthTransform()                   # After all the transforms, keep the same length
        )

def get_val_transformations(p):
    if p['augmentation_strategy'] == 'audio':
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