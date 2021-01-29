import torch
from torch.utils.data import WeightedRandomSampler



from data.datasets import SpeechCommands

dataset = SpeechCommands(num_labels=35, subset='training')
print(len(dataset))


import numpy as np

occurences = np.zeros(len(dataset.labels))

labels = []
for i, sample in enumerate(dataset):
    index = dataset.labels.index(sample['label'])
    occurences[index] += 1

probabilities = occurences/len(dataset)

# Reverse the probabilities
probabilities = (1 - probabilities)/sum(1-probabilities)

weights = np.zeros(len(dataset))
for i, sample in enumerate(dataset):
    index = dataset.labels.index(sample['label'])
    weights[i] = probabilities[index]
    #weights[i] = probabilities[index]/occurences[index]
    
batch_size = 30
samples = list(WeightedRandomSampler(weights, batch_size, replacement=True))
for sample in samples:
    print(dataset[sample]['label'])

# Weight tensor should contain a weight for each SAMPLE
