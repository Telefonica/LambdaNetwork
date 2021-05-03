import torch


def train_model(train_loader, model, criterion, optimizer):
    model.train()
    total_loss = 0

    for i, batch in enumerate(train_loader):
        
        input = batch['input'].cuda(non_blocking=True)
        
        output = model(input)
        target = batch['target']

        loss = criterion(output, target.cuda(non_blocking=True))
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print("Progress: {:.2f}% \t Batch: {} \t Loss: {}".format((i+1)/len(train_loader) * 100, (i+1), loss.item()))

    return total_loss / (i+1)


def validate_model(val_loader, model, criterion, optimizer):
    model.train()
    total_loss = 0

    for i, batch in enumerate(val_loader):
        
        input = batch['input'].cuda(non_blocking=True)

        output = model(input)
        target = batch['target']

        loss = criterion(output, target.cuda(non_blocking=True))
        total_loss += loss.item()

        # Do not compute these gradients
        optimizer.zero_grad()

    print("Validation Loss: {0}".format(total_loss/(i+1)))
    return total_loss / (i+1)


@torch.no_grad()
def test_model(test_loader, model):
    model.eval()
    corrects = 0
    total_samples = 0

    for i, batch in enumerate(test_loader):

        input = batch['input'].cuda(non_blocking=True)

        output = model(input)
        target = batch['target']

        batch_size = target.size(0)
        target_output = output.argmax(axis=1).to('cpu')
        
        corrects += torch.sum(target_output == target).item()
        total_samples += batch_size

    print('Accuracy of the model (Test Data): {0:.2f}%'.format(float(corrects/total_samples)*100))
    
    return float(corrects/total_samples)