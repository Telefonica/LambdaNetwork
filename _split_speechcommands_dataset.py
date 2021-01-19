from data.datasets import SpeechCommands

train_set = SpeechCommands(num_labels=12, subset="train")
val_set = SpeechCommands(num_labels=12, subset="validation")
test_set = SpeechCommands(num_labels=12, subset="test")

total_samples = len(train_set) + len(test_set) + len(val_set)
print('Total samples: {}'.format(total_samples))
print('Train samples: {} ({:.2f}%)'.format(len(train_set), len(train_set)/total_samples*100))
print('Val samples: {} ({:.2f}%)'.format(len(val_set), len(val_set)/total_samples*100))
print('Test samples: {} ({:.2f}%)'.format(len(test_set), len(test_set)/total_samples*100))