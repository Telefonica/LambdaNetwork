import numpy as np
import matplotlib.pyplot as plt

train_loss = np.load('output/lambdaResnet50_35/train_loss.npy')
val_loss = np.load('output/lambdaResnet50_35/val_loss.npy')

plt.plot(train_loss)
plt.plot(val_loss)

# Show the validation values (track some NaN)
print(val_loss)

plt.show()

test_acc = np.load('output/lambdaResnet50_35/test_accuracy.npy')
plt.plot(test_acc)
plt.show()