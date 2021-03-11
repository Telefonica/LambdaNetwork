import os
import glob
import numpy as np
import itertools
import matplotlib.pyplot as plt


def get_topk_table(indicies, filenames):

    # Get each value of the saved npy and assign a name
    topk_string = []
    for sample in indicies:
        list_topk = []
        for neighbor in sample:
            list_topk.append(filenames[neighbor])
        topk_string.append(list_topk)

    return topk_string


def save_confusion_matrix(cm, classes, path, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=70, fontsize=6)
    plt.yticks(tick_marks, classes, fontsize=6)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black", fontsize=4)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(path, dpi=1000)
