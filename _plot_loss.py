import numpy as np
import matplotlib.pyplot as plt

train_loss = np.load('output/google_commands/train_loss.npy')
val_loss = np.load('output/google_commands/val_loss.npy')

plt.plot(train_loss)
plt.plot(val_loss)

# Show the validation values (track some NaN)
print(val_loss)

plt.show()