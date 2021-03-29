import os
import matplotlib.pyplot as plt
import numpy as np

lambdaResnet15_path = 'output/google_commands/LambdaResnet15_1D_mel_35_iter3'
Resnet15_1d_path = 'output/google_commands/Resnet15_1D_mel_35_iter2'
Resnet15_2d_path = 'output/google_commands/Resnet15_2D_mel_35_iter3'
paths = [lambdaResnet15_path, Resnet15_1d_path, Resnet15_2d_path]
models = ['LambdaResnet15', 'Resnet15-1D', 'Resnet15-2D']

colors = ['orange', 'olive', 'brown']
linestyles = ['--', '-.', '-']

for i, path in enumerate(paths):
    fpr = np.load(os.path.join(path, 'fpr.npy'))
    tpr = np.load(os.path.join(path, 'tpr.npy'))
    auc = np.load(os.path.join(path, 'auc.npy'))

    plt.plot(fpr, tpr, label='{} (AUC: {:0.4f})'.format(models[i],auc), color=colors[i], linestyle=linestyles[i], linewidth=2)
    
plt.legend(loc="lower right")

plt.xlim(0, 0.1)
plt.ylim(0.9, 1)

plt.title('Receiver Operating Characteristic ROC')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.savefig('ROC.png', dpi=1000)
plt.show()