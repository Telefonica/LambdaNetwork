import numpy as np
import torch
from torch.utils.data import Dataset
import random
import torchaudio


'''
    MelDataset
    Returns an augmented audio (1D) and its MEL Spectogram (2D)
'''
class MelDataset(Dataset):
    def __init__(self, dataset, n_fft, n_mels, win_size, win_stride, sample_rate, transform):
        super(MelDataset, self).__init__()

        # Get the dataset transformation to be applied and block it's transformation
        self.dataset = dataset
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            win_length=int(win_size*sample_rate),
            hop_length=int(win_stride*sample_rate) )
        
        self.length = 16000
        self.transform = transform
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        audio, sample_rate, label, speaker_id, utterance_num = self.dataset.__getitem__(index)
        
        # If the label is not in the label list
        if label not in self.dataset.labels:
            label = 'unknown'

        # Audio data augmentation
        audio = self.transform(audio)

        # Getting the spectogram
        mel_spectogram = torch.log(self.mel(audio)+0.001)
        
        return {'audio': audio,
                'mel_spectogram': mel_spectogram,
                'label': label,
                'target': self.dataset.labels.index(label)}

""" 
    AugmentedDataset
    Returns an audio / mel together with an augmentation.
"""
class AugmentedDataset(Dataset):
    def __init__(self, dataset, transform):
        super(AugmentedDataset, self).__init__()

        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset.__getitem__(index)

        audio_aug = self.transform(sample['audio'])
        mel_spectogram_aug = torch.log(self.dataset.mel(audio_aug)+0.001)

        return {'audio': sample['audio'],
                'audio_aug': audio_aug,
                'mel_spectogram': sample['mel_spectogram'],
                'mel_spectogram_aug': mel_spectogram_aug,
                'label': sample['label'],
                'target': sample['target'] }

""" 
    NeighborsDataset
    Returns an image with one of its neighbors.
"""
class NeighborsDataset(Dataset):
    def __init__(self, dataset, indices, num_neighbors=None):
        super(NeighborsDataset, self).__init__()
        transform = dataset.transform
        
        if isinstance(transform, dict):
            self.anchor_transform = transform['standard']
            self.neighbor_transform = transform['augment']
        else:
            self.anchor_transform = transform
            self.neighbor_transform = transform
       
        dataset.transform = None
        self.dataset = dataset
        self.indices = indices # Nearest neighbor indices (np.array  [len(dataset) x k])
        if num_neighbors is not None:
            self.indices = self.indices[:, :num_neighbors+1]
        assert(self.indices.shape[0] == len(self.dataset))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        output = {}
        anchor = self.dataset.__getitem__(index)
        
        neighbor_index = np.random.choice(self.indices[index], 1)[0]
        neighbor = self.dataset.__getitem__(neighbor_index)

        anchor['image'] = self.anchor_transform(anchor['image'])
        neighbor['image'] = self.neighbor_transform(neighbor['image'])

        output['anchor'] = anchor['image']
        output['neighbor'] = neighbor['image'] 
        output['possible_neighbors'] = torch.from_numpy(self.indices[index])
        output['target'] = anchor['target']
        
        return output
