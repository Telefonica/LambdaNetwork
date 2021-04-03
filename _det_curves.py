import os
import matplotlib.pyplot as plt
import numpy as np

num_classes = 10

lambdaResnet18 = 'output/google_commands/LambdaResnet_1D_mel_{}_k1'.format(num_classes)
lambdaResnet18_2 = 'output/google_commands/LambdaResnet_1D_mel_{}_k2'.format(num_classes)
TC_Resnet14 = 'output/google_commands/Resnet_1D_mel_{}'.format(num_classes)
Resnet15 = 'output/google_commands/Resnet_2D_mel_{}'.format(num_classes)

paths = [Resnet15, lambdaResnet18_2, lambdaResnet18, TC_Resnet14]
models = ['Resnet15', 'LambdaResnet18-2', 'LambdaResnet18', 'TC-Resnet14']

colors = ['cornflowerblue', 'darkorange', 'brown', 'mediumseagreen']
linestyles = ['--', '-', '-.', ':']

for i, path in enumerate(paths):
    fpr = np.load(os.path.join(path, 'fpr_det.npy'))
    tpr = np.load(os.path.join(path, 'tpr_det.npy'))
    auc = np.load(os.path.join(path, 'auc_det.npy'))

    plt.plot(fpr, tpr, label='{} (AUC: {:0.3f}e-03)'.format(models[i], auc*1000), color=colors[i], linestyle=linestyles[i], linewidth=1.5)
    
plt.legend(loc="upper right")

plt.xlim(0, 0.08)
plt.ylim(0, 0.08)

#plt.title('Receiver Operating Characteristic ROC')
plt.xlabel('False alarm rate (false positive)')
plt.ylabel('False reject rate (false negative)')

plt.savefig('DET_{}.png'.format(num_classes), dpi=1000, bbox_inches='tight')
plt.show()