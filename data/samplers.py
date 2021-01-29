import torch
from torch.utils.data import WeightedRandomSampler
import numpy as np

def get_SpeechCommandsSampler(p, dataset):
    occurences = np.zeros(len(dataset.labels))

    # Get the occurences of each class
    print("Computing the occurences of each class in the dataset ...")
    for i, sample in enumerate(dataset.dataset):
        index = dataset.labels.index(sample['label'])
        occurences[index] += 1

    probabilities = occurences/len(dataset)

    # Reverse the probabilities
    probabilities = (1-probabilities)/sum(1-probabilities)

    print("Computing the weight of each sampler for the Weighted sampler ...")
    # Get the weight for each sample in the dataset
    weights = np.zeros(len(dataset))
    for i, sample in enumerate(dataset.dataset):
        index = dataset.labels.index(sample['label'])
        weights[i] = probabilities[index]
        #weights[i] = probabilities[index]/occurences[index]
    
    return WeightedRandomSampler(weights, len(dataset), replacement=True)
