import numpy as np
import torch
from torch.utils.data import Dataset
import random
import torchaudio


'''
    AudioDataset
    Returns an aumgneted audio (1D)
'''
class AudioDataset(Dataset):
    def __init__(self, dataset, transform):
        super(AudioDataset, self).__init__()

        # Get the dataset transformation to be applied and block it's transformation
        self.dataset = dataset
        self.transform = transform
        self.labels = dataset.labels
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset.__getitem__(index)
        
        # If the label is not in the label list
        label = sample['label']
        if label not in self.labels:
            label = 'unknown'
        
        # Audio data augmentation
        audio = self.transform(sample['audio'])
        
        return {'input': audio,
                'label': label,
                'target': self.labels.index(label)}


'''
    MelDataset
    Returns the MEL Spectogram (2D) of an augmented audio
'''
class MelDataset(Dataset):
    def __init__(self, dataset, n_fft, n_mels, win_size, win_stride, sample_rate):
        super(MelDataset, self).__init__()

        # Get the dataset transformation to be applied and block it's transformation
        self.dataset = dataset
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            win_length=int(win_size*sample_rate),
            hop_length=int(win_stride*sample_rate) )
        
        self.labels = dataset.labels
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset.__getitem__(index)
        
        # If the label is not in the label list
        label = sample['label']
        if label not in self.labels:
            label = 'unknown'
        
        # Getting the spectogram
        mel_spectogram = torch.log(self.mel(sample['input'])+0.001)
        
        return {'input': mel_spectogram,
                'label': label,
                'target': self.labels.index(label)}
