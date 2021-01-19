from torchaudio.datasets import SPEECHCOMMANDS
import os
import glob

class SpeechCommands(SPEECHCOMMANDS):
    def __init__(self, num_labels: int = 35, subset: str = None):
        super().__init__('./datasets/', url='speech_commands_v0.02', folder_in_archive='SpeechCommands', download=True)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.join(self._path, line.strip()) for line in fileobj]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "test":
            self._walker = load_list("testing_list.txt")
        elif subset == "train":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]

        self.labels = []
        if num_labels == 12:
            self.labels = sorted(['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'unknown'])
        else:
            # Get the labels in form of the folder
            for folder in glob.glob(self._path + '/*/'):
                folder = folder.split('/')
                self.labels.append(folder[len(folder)-2])
            
            # Replace the background noise for the unknown label:
            self.labels[self.labels.index('_background_noise_')] = 'unknown'
            self.labels = sorted(self.labels)