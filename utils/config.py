"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os
import yaml
from easydict import EasyDict

def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

def create_config(config_file_env, config_file_exp):
    # Config for environment path
    with open(config_file_env, 'r') as stream:
        root_dir = yaml.safe_load(stream)['root_dir']
   
    with open(config_file_exp, 'r') as stream:
        config = yaml.safe_load(stream)
    
    cfg = EasyDict()
   
    # Copy
    for k, v in config.items():
        cfg[k] = v

    # Set paths for the task of the dataset:
    dataset_dir = os.path.join(root_dir, cfg['db_name'])
    mkdir_if_missing(dataset_dir)

    model_name = cfg['backbone'] + '_' + cfg['setup'] + '_' + cfg['frontend'] + '_' + str(cfg['num_labels'])
    base_dir = os.path.join(dataset_dir, model_name)
    mkdir_if_missing(base_dir)
    
    cfg['base_dir'] = base_dir
    cfg['checkpoint_dir'] = os.path.join(base_dir, 'checkpoint.pth.tar')
    cfg['model_test'] = os.path.join(base_dir, 'best_model.pth.tar')
    cfg['model_dir'] = os.path.join(base_dir, 'model.pth.tar')
    cfg['train_loss_dir'] = os.path.join(base_dir, 'train_loss.npy')
    cfg['val_loss_dir'] = os.path.join(base_dir, 'val_loss.npy')
    cfg['test_acc_dir'] = os.path.join(base_dir, 'test_accuracy.npy')

    return cfg 
