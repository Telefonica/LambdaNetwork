import os
import matplotlib.pyplot as plt
import numpy as np

num_classes = 10

lambdaResnet18 = 'output/google_commands/LambdaResnet_{}_k1'.format(num_classes)
lambdaResnet18_2 = 'output/google_commands/LambdaResnet_{}_k2'.format(num_classes)
TC_Resnet14 = 'output/google_commands/Resnet_{}'.format(num_classes)
Resnet15 = 'output/google_commands/Resnet_{}'.format(num_classes)

paths = [Resnet15, lambdaResnet18_2, lambdaResnet18, TC_Resnet14]
models = ['Resnet15', 'LambdaResnet18-2', 'LambdaResnet18', 'TC-Resnet14']

colors = ['cornflowerblue', 'darkorange', 'brown', 'mediumseagreen']
linestyles = ['--', '-', '-.', ':']

for i, path in enumerate(paths):
    fpr = np.load(os.path.join(path, 'fpr_roc.npy'))
    tpr = np.load(os.path.join(path, 'tpr_roc.npy'))
    auc = np.load(os.path.join(path, 'auc_roc.npy'))

    plt.plot(fpr, tpr, label='{} (AUC: {:0.4f})'.format(models[i],auc), color=colors[i], linestyle=linestyles[i], linewidth=1.5)
    
plt.legend(loc="lower right")

plt.xlim(0, 0.1)
plt.ylim(0.9, 1)

#plt.title('Receiver Operating Characteristic ROC')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

#plt.savefig('ROC_{}.png'.format(num_classes), dpi=1000, bbox_inches='tight')
plt.show()