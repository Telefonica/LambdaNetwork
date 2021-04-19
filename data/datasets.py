import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS
from torch.utils.data import Dataset
from torchaudio import load
import os
import glob
import csv

class SpeechCommands(SPEECHCOMMANDS):
    def __init__(self, num_labels: int = 35, subset: str = None):
        if torchaudio.__version__.startswith('0.7'):
            super().__init__('./datasets/', url='speech_commands_v0.02', folder_in_archive='SpeechCommands', download=True)

            def load_list(filename):
                filepath = os.path.join(self._path, filename)
                with open(filepath) as fileobj:
                    return [os.path.join(self._path, line.strip()) for line in fileobj]

            if subset == "validation":
                self._walker = load_list("validation_list.txt")
            elif subset == "testing":
                self._walker = load_list("testing_list.txt")
            elif subset == "training":
                excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
                excludes = set(excludes)
                self._walker = [w for w in self._walker if w not in excludes]

        elif torchaudio.__version__.startswith('0.8'):
            super().__init__(root='./datasets/', url='speech_commands_v0.02', folder_in_archive='SpeechCommands', download=True, subset=subset)
        
        else:
            print('Unsuported torchaudio version for the Google Speech Comands dataset')
            import sys
            sys.exit()
        
        self.labels = ['unknown']
        if num_labels == 2:
            self.labels.extend(['left', 'right'])
        
        elif num_labels == 10:
            self.labels.extend(['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go'])
        
        elif num_labels == 20:
            self.labels.extend(['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go',
                                'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'])
        else:
            # Get the labels in form of the folder
            for folder in glob.glob(self._path + '/*/'):
                folder = folder.split('/')
                self.labels.append(folder[len(folder)-2])
            
            # Remove background noise and unknown (whole dataset used):
            self.labels.remove('_background_noise_')
            self.labels.remove('unknown')
        
        self.labels = sorted(self.labels)

    def __getitem__(self, index):
        audio, sample_rate, label, speaker_id, utterance_number = super().__getitem__(index)

        # If the label is not in the label list
        if label not in self.labels:
            label = 'unknown'
        
        return {'audio': audio,
                'sample_rate': sample_rate,
                'label': label }


class TAUurban(Dataset):
    def __init__(self, dataset_path='datasets/TAU-urban-acoustic-scenes-2020-mobile-development', num_labels: int = 10, subset: str = None):

        if subset == 'training':
            files_csv = os.path.join(dataset_path, 'evaluation_setup/fold1_train.csv')
        elif subset == 'testing':
            files_csv = os.path.join(dataset_path, 'evaluation_setup/fold1_test.csv')
        elif subset == 'validation':
            files_csv = os.path.join(dataset_path, 'evaluation_setup/fold1_evaluate.csv')
        
        with open(files_csv, newline='') as f:
            reader = csv.reader(f)
            data = list(reader)

        self.files = []
        self.files_labels = [] 

        # Skip the title
        for wavfile in data[1:]:
            filename = wavfile[0].split('\t')
            
            # Some samples have their label
            if len(filename) == 2:
                filename, label = filename
            elif len(filename) == 1:
                filename = filename[0]
                label = os.path.basename(filename).split('-')[0]

            self.files.append(os.path.join(dataset_path, filename))
            self.files_labels.append(label)
        
        self.labels = sorted(set(self.files_labels))

    def __len__(self):
        return len(self.files_labels)

    def __getitem__(self, index):

        # Load the audio file
        audio, sample_rate = load(self.files[index])
        label = self.files_labels[index]

        return {'audio': audio,
                'sample_rate': sample_rate,
                'label': label }

