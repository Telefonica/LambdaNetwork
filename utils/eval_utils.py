import numpy as np
import glob
import os


def get_topk_table(indicies, filenames):

    # Get each value of the saved npy and assign a name
    topk_string = []
    for sample in indicies:
        list_topk = []
        for neighbor in sample:
            list_topk.append(filenames[neighbor])
        topk_string.append(list_topk)

    return topk_string