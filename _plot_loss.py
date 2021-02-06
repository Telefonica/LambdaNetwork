import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Helper function to visualize loss and accuracy')
parser.add_argument('--output', help='Output folder to evaluare')
args = parser.parse_args()

# Get the str variables from the folder
model, keywords = args.output.split('/')[1].split('_')

# Load the files
train_loss = np.load(args.output + 'train_loss.npy')
val_loss = np.load(args.output + 'val_loss.npy')
test_acc = np.load(args.output + 'test_accuracy.npy')

# Two plots: Loss and Accuracy
fig, (loss_axs, acc_axs) = plt.subplots(1,2,figsize=(15, 6))
fig.suptitle(model + ' ' + keywords + ' Keywords', fontsize=14)

# Loss plot
loss_axs.title.set_text('Loss evolution')
loss_axs.plot(train_loss, label='Train')
loss_axs.plot(val_loss, label='Validation')
loss_axs.legend(loc='upper right')

loss_axs.set_xlabel('Epoch')
loss_axs.set_ylabel('Loss value')

loss_axs.grid(linestyle='dotted')

# Accuracy plot
acc_axs.title.set_text('Accuracy evolution')
acc_axs.plot(test_acc, label='Test')

acc_axs.set_ylim([0.,1.])
acc_axs.set_yticks(np.arange(0, 1.01, 0.1))

y_max_acc = max(test_acc)
x_max_acc = np.where(test_acc == y_max_acc)[0].item()
acc_axs.scatter(x_max_acc, y_max_acc, color='r', label="Max Acc: {:.2f}%".format(y_max_acc * 100))

acc_axs.legend(loc='lower right')

acc_axs.set_xlabel('Epoch')
acc_axs.set_ylabel('Accuracy')

acc_axs.grid(linestyle='dotted')

# Show the resuling plot
plt.show()
