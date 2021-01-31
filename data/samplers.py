import torch
from torch.utils.data import WeightedRandomSampler
import numpy as np

def get_SpeechCommandsSampler(p, dataset):
    num_labels = len(dataset.labels)
    occurences = np.zeros(num_labels)

    # Get the occurences of each class
    print("Computing the occurences of each class in the dataset ...")
    for i, sample in enumerate(dataset.dataset):
        index = dataset.labels.index(sample['label'])
        occurences[index] += 1

    probabilities = occurences/len(dataset)
    wieghts = 1/(num_labels*probabilities)

    print("Computing the weight of each sampler for the Weighted sampler ...")
    # Get the weight for each sample in the dataset
    weights_sample = np.zeros(len(dataset))
    for i, sample in enumerate(dataset.dataset):
        index = dataset.labels.index(sample['label'])
        weights_sample[i] = wieghts[index]
    
    return WeightedRandomSampler(weights_sample, len(dataset), replacement=True)
