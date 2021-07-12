import os
import torch
from torch.utils.data import WeightedRandomSampler
import numpy as np

def get_SpeechCommandsSampler(p, dataset):
    num_labels = len(dataset.labels)
    occurences = np.zeros(num_labels)
    idx_sample = np.zeros(len(dataset))

    weight_file = os.path.join(p['dataset_dir'], 'probs_{}.npy'.format(p['num_labels']))

    if os.path.exists(weight_file):
        weights_sample = np.load(weight_file)

    else:
        # Get the occurences of each class
        print("Computing the occurences of each class in the dataset ({})...".format(p['num_labels']))
        for i, sample in enumerate(dataset.dataset):
            index = dataset.labels.index(sample['label'])
            occurences[index] += 1
            idx_sample[i] = index

            if i % 10000 == 0:
                print('({:.2f}%) Evaluated {}/{} samples'.format((i/len(dataset)*100), i, len(dataset)))

        weights = len(dataset)/occurences
        print('Class weights: {}'.format(weights))

        weights_sample = np.zeros(len(dataset))
        for i in range(0, num_labels):
            weights_sample[np.where(idx_sample==i)] = weights[i]
            
        np.save(weight_file, weights_sample)
    
    assert (len(weights_sample) == len(dataset))
    
    return WeightedRandomSampler(weights_sample, len(dataset), replacement=False)
