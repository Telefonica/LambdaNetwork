import os
import yaml
from easydict import EasyDict

def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def create_config(config_file_exp):
    
    with open(config_file_exp, 'r') as stream:
        config = yaml.safe_load(stream)
    
    cfg = EasyDict()
    # Copy
    for k, v in config.items():
        cfg[k] = v

    root_dir = 'output/'
    # Set paths for the task of the dataset:
    dataset_dir = os.path.join(root_dir, cfg['db_name'])
    mkdir_if_missing(dataset_dir)
    cfg['dataset_dir'] = dataset_dir
    
    model_name = cfg['backbone'] + '_' + str(cfg['num_labels'])
    base_dir = os.path.join(dataset_dir, model_name)
    mkdir_if_missing(base_dir)
    
    cfg['base_dir'] = base_dir
    cfg['checkpoint_dir'] = os.path.join(base_dir, 'checkpoint.pth.tar')
    cfg['best_model_dir'] = os.path.join(base_dir, 'best_model.pth.tar')
    cfg['model_dir'] = os.path.join(base_dir, 'model.pth.tar')
    cfg['train_loss_dir'] = os.path.join(base_dir, 'train_loss.npy')
    cfg['val_loss_dir'] = os.path.join(base_dir, 'val_loss.npy')
    cfg['test_acc_dir'] = os.path.join(base_dir, 'test_accuracy.npy')

    return cfg 
