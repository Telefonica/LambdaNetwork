import argparse
import torch
import numpy as np

from utils.config import create_config
from utils.train_utils import simclr_train, supervised_train, supervised_val, supervised_test
from utils.common_config import get_model, get_criterion, get_optimizer, adjust_learning_rate, get_dataset,\
                                get_train_transformations, get_train_dataloader,\
                                get_val_transformations, get_val_dataloader

from torchsummary.torchsummary import summary

parser = argparse.ArgumentParser(description='Lambda ResNet')
parser.add_argument('--config_env', help='Config file for environment')
parser.add_argument('--config_exp', help='Config file for experiment')
parser.add_argument('--gpu', help='GPU to use')
args = parser.parse_args()

def main():
    p = create_config(args.config_env, args.config_exp)

    cuda = torch.cuda.is_available()

    model = get_model(p)
    print('Model is {}'.format(model.__class__.__name__))
    
    # Use Data parallel mode if not GPU is selected through the args
    if cuda:
        if args.gpu is None:
            device = torch.device('cuda:0')
            model = torch.nn.DataParallel(model)
            p['batch_size'] = torch.cuda.device_count() * p['batch_size']
        else:
            device = torch.device('cuda:{}'.format(args.gpu))
            torch.cuda.set_device(device)
        model.cuda()
    
    # Dataset: Get Train and Evaluation
    train_transforms = get_train_transformations(p)
    val_transforms = get_val_transformations(p)
    
    # Split whole dataset in train/validation and get loaders
    if p['setup'] == 'contrastive':
        to_augmented_dataset = True
    elif p['setup'] == 'supervised':
        to_augmented_dataset = False

    train_dataset = get_dataset(p, train_transforms, to_augmented_dataset=to_augmented_dataset, subset="training")
    val_dataset = get_dataset(p, val_transforms, to_augmented_dataset=False, subset="validation")
    test_dataset = get_dataset(p, val_transforms, to_augmented_dataset=False, subset="testing")
    
    train_dataloader = get_train_dataloader(p, train_dataset)
    val_dataloader = get_val_dataloader(p, val_dataset)
    test_dataloader = get_val_dataloader(p, test_dataset)

    # Print the model shape and memory on the GPU
    if cuda:
        print(summary(model, train_dataset[0]['mel_spectogram'].cuda().shape, batch_size=p['batch_size'], device=device))
    else:
        print(summary(model, train_dataset[0]['mel_spectogram'].shape, batch_size=p['batch_size'], device='cpu'))

    print('Train transforms:', train_transforms)
    print('Validation transforms:', val_transforms)
    print("Dataset contains {}/{} train/val samples".format(len(train_dataset),len(val_dataset)))

    # Criterion
    criterion = get_criterion(p)
    print('Criterion is {}'.format(criterion.__class__.__name__))
    if cuda:
        criterion = criterion.cuda()

    # Optimizer and scheduler
    optimizer = get_optimizer(p, model)
    print(optimizer)

    # TODO: Checkpoint load the last epoch
    start_epoch = 0

    print("Starting the Training loop ...")
    print("    Number of Epochs: {}".format(p['epochs']))
    print("    Batch size: {}".format(p['batch_size']))
    print("    Total batches: {:.2f}".format(len(train_dataset)/p['batch_size']))

    loss_train_values = []
    loss_val_values = []
    acc_test_values = []
    current_best_acc = 0.0

    for epoch in range(start_epoch, p['epochs']):
        print("Epoch {}/{}".format(epoch, p['epochs']))

        # Adjust lr
        lr = adjust_learning_rate(p, optimizer, epoch)
        print('Adjusted learning rate to {:.5f}'.format(lr))

        # Train
        print('Train...')
        if p['setup'] == 'contrastive':
            loss_train = simclr_train(train_dataloader, model, criterion, optimizer, epoch)
            loss_val = 0
        elif p['setup'] == 'supervised':
            loss_train = supervised_train(train_dataloader, model, criterion, optimizer)
            loss_val = supervised_val(val_dataloader, model, criterion, optimizer)
            acc_test = supervised_test(test_dataloader, model, criterion)

            if current_best_acc < acc_test:
                print('Saving the most accurate current model for the Test dataset')
                torch.save(model.state_dict(), p['model_test'])
                current_best_acc = acc_test

        loss_train_values.append(loss_train)
        loss_val_values.append(loss_val)
        acc_test_values.append(acc_test)

        # Checkpoint
        print('Checkpoint ...')
        torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(), 
                    'epoch': epoch + 1}, p['checkpoint_dir'])
    
    # Save final model
    torch.save(model.state_dict(), p['model_dir'])

    # Save the training loss values
    np.save(p['train_loss_dir'], loss_train_values)
    np.save(p['val_loss_dir'], loss_val_values)
    np.save(p['test_acc_dir'], acc_test_values)

    print('Training complete')

if __name__ == "__main__":
    main()