import os
import glob
import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, det_curve, auc


def save_confusion_matrix(cm, classes, path, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

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


def get_roc_curve(y_true, probs, class_idx=None):
    n_classes = probs.shape[1]
    print(probs.shape)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    y_true_all = np.zeros(probs.shape)
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(np.equal(y_true, i).astype(int), probs[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        y_true_all[:, i] = np.equal(y_true, i).astype(int)

    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_all.ravel(), probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    if class_idx is None:
        return fpr['micro'], tpr['micro'], roc_auc['micro']
    else:
        return fpr[i], tpr[i], roc_auc[i]


def get_det_curve(y_true, probs, class_idx=None):
    n_classes = probs.shape[1]
    print(probs.shape)
    fpr = dict()
    fnr = dict()
    roc_auc = dict()
    
    y_true_all = np.zeros(probs.shape)
    
    for i in range(n_classes):
        print(np.equal(y_true, i).astype(int))
        fpr[i], fnr[i], _ = det_curve(np.equal(y_true, i).astype(int), probs[:,i])
        roc_auc[i] = auc(fpr[i], fnr[i])
        y_true_all[:, i] = np.equal(y_true, i).astype(int)

    fpr["micro"], fnr["micro"], _ = det_curve(y_true_all.ravel(), probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], fnr["micro"])
    
    if class_idx is None:
        return fpr['micro'], fnr['micro'], roc_auc['micro']
    else:
        return fpr[i], fnr[i], roc_auc[i]