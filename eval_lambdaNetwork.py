import argparse
import torch
import yaml
from tabulate import tabulate
import numpy as np
from utils.common_config import get_dataset, get_val_transformations, get_val_dataloader, get_model
from utils.eval_utils import get_topk_table

cuda = torch.cuda.is_available()
if cuda:
    from utils.memory import MemoryBank, fill_memory_bank

FLAGS = argparse.ArgumentParser(description='Evaluate models from the model zoo')
FLAGS.add_argument('--config_exp', help='Location of config file')
FLAGS.add_argument('--model', help='Location where model is saved')
FLAGS.add_argument('--gpu', help='GPU device to use')
args = FLAGS.parse_args()

def main():
    
    # Read config file
    print('Read config file {} ...'.format(args.config_exp))
    with open(args.config_exp, 'r') as stream:
        config = yaml.safe_load(stream)

    # Get model
    print('Getting the model')
    model = get_model(config)

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
    val_transforms = get_val_transformations(config)
    test_dataset = get_dataset(config, val_transforms, to_augmented_dataset=False, subset='test')
    test_dataloader = get_val_dataloader(config, test_dataset)
    print('Number of samples: {}'.format(len(test_dataset)))

    
    # Read model weights
    print('Load model weights ...')
    state_dict = torch.load(args.model, map_location='cpu')
    model.load_state_dict(state_dict)
    
    # CUDA
    if cuda:
        model.cuda()

    # Perform evaluation
    print('Perform evaluation of the task (setup={}).'.format(config['setup']))

    # TODO: Differentiate memroy bank of labels than memory bank of features (for unsupervised)
    if config['setup'] == 'supervised':

        print('Create Memory Bank')
        memory_bank = MemoryBank(len(test_dataset),
                                config['model_kwargs']['num_labels']+1,
                                config['num_labels']+1, config['criterion_kwargs']['temperature'])
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
        print('Accuracy: {} ({}/{})'.format(accuracy*100, corrects, len(test_dataset)))
        
    elif config['setup'] == 'contrastive':
        print('Create Memory Bank')
        memory_bank = MemoryBank(len(test_dataset),
                                config['model_kwargs']['features_dim'],
                                config['num_events'], config['criterion_kwargs']['temperature'])
        if cuda:
            memory_bank.cuda()

        print('Mine the nearest neighbors')
    
        # Mine the nearest neighbours of the learned features
        for topk in [2, 5]: 
            indicies = memory_bank.mine_nearest_neighbors(topk)
            labels = []
            for data in test_dataset:
                labels.append(data['label'])
            table_str = get_topk_table(indicies, labels)

            # Print the table
            headers = ['Keyword']
            headers.extend('Top ' + str(k+1) for k in range(0,topk))
            print("Top k={} nearest neighbors".format(topk))
            print(tabulate(table_str, headers=headers))

            # Check match
            correct = 0
            for sample in table_str:
                if sample[0] == sample[1]:
                    correct = correct + 1
            print(correct)
            print('Accuracy: {}/{}'.format(correct, len(test_dataset)))

        print("Features Evaluated saved")

if __name__ == "__main__":
    main() 
