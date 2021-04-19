import glob
import os
import csv

dataset_path='datasets/TAU-urban-acoustic-scenes-2020-mobile-development'

waviles_path = os.path.join(dataset_path, 'evaluation_setup/fold1_train.csv')
print(waviles_path)

with open(waviles_path, newline='') as f:
    reader = csv.reader(f)
    data = list(reader)

filenames = []
labels = [] 

for wavfile in data[1:]:
    filename, label = wavfile[0].split('\t')
    filenames.append(os.path.join(dataset_path, filename))
    labels.append(label)

filename = filenames[0]
print(filename)

from torchaudio import load, save
audio, sample_rate = load(filename)

print(audio.shape)

save('saving.wav', audio, sample_rate)