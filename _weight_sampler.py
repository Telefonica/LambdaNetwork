import torch
from torch.utils.data import WeightedRandomSampler
from data.datasets import SpeechCommands
import numpy as np

dataset = SpeechCommands(num_labels=2, subset='training')
print(len(dataset))

occurences = np.zeros(len(dataset.labels))
num_labels = len(dataset.labels)

labels = []
for i, sample in enumerate(dataset):
    index = dataset.labels.index(sample['label'])
    occurences[index] += 1

probabilities = occurences/len(dataset)

# Reverse the probabilities
weights = 1 / (num_labels * probabilities)

weights_sample = np.zeros(len(dataset))
for i, sample in enumerate(dataset):
    index = dataset.labels.index(sample['label'])
    weights_sample[i] = weights[index]
    #weights[i] = probabilities[index]/occurences[index]
    
batch_size = 30
samples = list(WeightedRandomSampler(weights_sample, batch_size, replacement=True))
for sample in samples:
    print(dataset[sample]['label'])