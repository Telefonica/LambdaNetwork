import argparse
import torch
import numpy as np
import os

from utils.config import create_config
from utils.train_utils import train_model, validate_model, test_model
from utils.common_config import get_model, get_criterion, get_optimizer, adjust_learning_rate, get_dataset,\
                                get_train_transformations, get_train_dataloader,\
                                get_val_transformations, get_val_dataloader

from torchsummary.torchsummary import summary

parser = argparse.ArgumentParser(description='Setup for training a keyword spotting task')
parser.add_argument('--config_exp', help='Config file for experiment')
parser.add_argument('--gpu', help='GPU to use')
args = parser.parse_args()

def main():
    p = create_config(args.config_exp)

    cuda = torch.cuda.is_available()

    model = get_model(p)
    print('Model is {}'.format(model.__class__.__name__))
    
    # Use Data parallel mode if not GPU is selected through the args
    if cuda:
        if args.gpu is None:
            device = torch.device('cuda:0')
            model = torch.nn.DataParallel(model)
            #p['batch_size'] = torch.cuda.device_count() * p['batch_size']
        else:
            device = torch.device('cuda:{}'.format(args.gpu))
            torch.cuda.set_device(device)
        model.cuda()
    
    # Get the datasets, transformations and dataloaders
    train_transforms = get_train_transformations(p)
    val_transforms = get_val_transformations(p)

    train_dataset = get_dataset(p, train_transforms, subset="training")
    val_dataset = get_dataset(p, val_transforms, subset="validation")
    test_dataset = get_dataset(p, val_transforms, subset="testing")
    
    train_dataloader = get_train_dataloader(p, train_dataset)
    val_dataloader = get_val_dataloader(p, val_dataset)
    test_dataloader = get_val_dataloader(p, test_dataset)
    
    if cuda:
        print(summary(model, train_dataset[0]['input'].cuda().shape, batch_size=p['batch_size'], device=device))
    else:
        print(summary(model, train_dataset[0]['input'].shape, batch_size=p['batch_size'], device='cpu'))
    
    print('Train transforms:', train_transforms)
    print('Validation transforms:', val_transforms)
    print('Dataset contains {}/{} train/val samples'.format(len(train_dataset),len(val_dataset)))

    # Criterion
    criterion = get_criterion(p)
    print('Criterion is {}'.format(criterion.__class__.__name__))
    if cuda:
        criterion = criterion.cuda()

    # Optimizer and scheduler
    optimizer = get_optimizer(p, model)
    print(optimizer)

    # TCheckpoint load the last epoch
    if os.path.exists(p['checkpoint_dir']):
        print('Restart from checkpoint {}'.format(p['checkpoint_dir']))
        
        checkpoint = torch.load(p['checkpoint_dir'], map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])        
        
        start_epoch = checkpoint['epoch']
        loss_train_values = np.load(p['train_loss_dir']).tolist()
        loss_val_values = np.load(p['val_loss_dir']).tolist()
        acc_test_values = np.load(p['test_acc_dir']).tolist()
        current_best_acc = checkpoint['best_acc']
    
    else:
        print('No checkpoint file at {}'.format(p['checkpoint_dir']))
        start_epoch = 0
        loss_train_values = []
        loss_val_values = []
        acc_test_values = []
        current_best_acc = 0.0
        
    print("Starting the Training loop ...")
    print("\tNumber of Epochs: {}".format(p['epochs']))
    print("\tBatch size: {}".format(p['batch_size']))
    print("\tTotal batches: {:.2f}".format(len(train_dataset)/p['batch_size']))

    for epoch in range(start_epoch, p['epochs']):
        print("Epoch {}/{}".format(epoch, p['epochs']))

        # Adjust lr
        lr = adjust_learning_rate(p, optimizer, epoch)
        print('Adjusted learning rate to {:.5f}'.format(lr))

        # Train
        print('Train...')
        loss_train = train_model(train_dataloader, model, criterion, optimizer)
        loss_val = validate_model(val_dataloader, model, criterion, optimizer)
        acc_test = test_model(test_dataloader, model)

        if current_best_acc < acc_test:
            print('Saving the most accurate model ...')
            torch.save(model.state_dict(), p['best_model_dir'])
            current_best_acc = acc_test

        loss_train_values.append(loss_train)
        loss_val_values.append(loss_val)
        acc_test_values.append(acc_test)

        # Checkpoint
        print('Checkpoint ...')
        torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(), 
                    'epoch': epoch + 1, 'best_acc': current_best_acc}, p['checkpoint_dir'])

        # Save the training loss values
        np.save(p['train_loss_dir'], loss_train_values)
        np.save(p['val_loss_dir'], loss_val_values)
        np.save(p['test_acc_dir'], acc_test_values)
    
    # Save final model
    torch.save(model.state_dict(), p['model_dir'])
    print('Training complete')

if __name__ == "__main__":
    main()