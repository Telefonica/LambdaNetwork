import argparse
import os
import torch
import yaml
import numpy as np
from sklearn.metrics import confusion_matrix
from utils.common_config import get_dataset, get_val_transformations, get_val_dataloader, get_model
from utils.eval_utils import get_topk_table
from utils.memory import MemoryBank, fill_memory_bank
from utils.eval_utils import save_confusion_matrix
from utils.config import create_config

parser = argparse.ArgumentParser(description='Evaluate Lambda ResNet')
parser.add_argument('--config_exp', help='Location of config file')
parser.add_argument('--model', help='Location where model is saved')
parser.add_argument('--gpu', help='GPU device to use')
parser.add_argument('--config_env', help='Config file for environment')
args = parser.parse_args()

def main():
    
    # Read config file
    p = create_config(args.config_env, args.config_exp)
    cuda = torch.cuda.is_available()

    # Get model
    print('Getting the model')
    model = get_model(p)

    # Use Data parallel mode if not GPU is selected through the args
    if cuda:
        if args.gpu is None:
            model = torch.nn.DataParallel(model)
            #p['batch_size'] = torch.cuda.device_count() * p['batch_size']
        else:
            device = torch.device('cuda:{}'.format(args.gpu))
            torch.cuda.set_device(device)
        model.cuda()

    # Get dataset
    print('Get validation dataset ...')
    val_transforms = get_val_transformations(p)
    test_dataset = get_dataset(p, val_transforms, subset='testing')
    test_dataloader = get_val_dataloader(p, test_dataset)
    print('Number of samples: {}'.format(len(test_dataset)))

    # Read model weights
    print('Load model weights ...')
    state_dict = torch.load(args.model, map_location='cpu')
    model.load_state_dict(state_dict)
    
    # CUDA
    if cuda:
        model.cuda()

    # Perform evaluation
    print('Perform evaluation of the task (setup={}).'.format(p['setup']))

    # TODO: Differentiate memroy bank of labels than memory bank of features (for unsupervised
    print('Create Memory Bank')
    memory_bank = MemoryBank(len(test_dataset),
                            p['model_kwargs']['num_labels'],
                            p['num_labels'], p['criterion_kwargs']['temperature'])
    if cuda:
        memory_bank.cuda()

    print('Fill Memory Bank')
    fill_memory_bank(test_dataloader, model, memory_bank)
    eval_output, eval_target = memory_bank.get_memory()

    print('Evaluating the predictions')
    eval_labels = eval_output.argmax(axis=1)
    corrects = 0
    for i in range(0, len(eval_labels)):
        if eval_labels[i].astype(int) == eval_target[i].astype(int):
            corrects += 1

    accuracy = corrects / float(len(eval_labels))

    matrix = confusion_matrix(eval_target, eval_labels)
    save_confusion_matrix(matrix, test_dataset.labels, os.path.join(p['base_dir'], 'confusion_matrix.png'))
    
    np.set_printoptions(threshold=np.inf)
    print('Confusion matrix: ')
    print(matrix)
    print('Accuracy: {} ({}/{})'.format(accuracy*100, corrects, len(test_dataset)))

if __name__ == "__main__":
    main() 
