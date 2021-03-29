from torchaudio.datasets import SPEECHCOMMANDS
import os
import glob

class SpeechCommands(SPEECHCOMMANDS):
    def __init__(self, num_labels: int = 35, subset: str = None):
        #super().__init__(root='./datasets/', url='speech_commands_v0.02', folder_in_archive='SpeechCommands', download=True, subset=subset)
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
