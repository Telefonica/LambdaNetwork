import numpy as np
import torch

class MemoryBank(object):
    def __init__(self, n, dim, num_classes):
        self.n = n
        self.dim = dim 
        
        if dim != 35:
            self.features = torch.FloatTensor(self.n, self.dim+1)
        else:
            self.features = torch.FloatTensor(self.n, self.dim)
        
        self.features = torch.FloatTensor(self.n, self.dim)
        self.targets = torch.FloatTensor(self.n)
        self.ptr = 0
        self.device = 'cpu'
        self.C = num_classes
        self.labels = []

    def reset(self):
        self.ptr = 0 
        
    def update(self, features, targets, label):
        b = features.size(0)
        assert(b + self.ptr <= self.n)
        self.features[self.ptr:self.ptr+b].copy_(features.detach())
        self.targets[self.ptr:self.ptr+b].copy_(targets.detach())
        self.labels.append(label)
        self.ptr += b

    def get_memory(self):
        return self.features.cpu().numpy(), self.targets.cpu().numpy()

    def to(self, device):
        self.features = self.features.to(device)
        self.device = device

    def cpu(self):
        self.to('cpu')

    def cuda(self):
        self.to('cuda:0')

@torch.no_grad()
def fill_memory_bank(loader, model, memory_bank):
    model.eval()
    memory_bank.reset()

    for i, batch in enumerate(loader):
        
        input = batch['input'].cuda(non_blocking=True)
        output = model(input)
        target = batch['target']
        label = batch['label']
        memory_bank.update(output, target, label)
        
        if i % 100 == 0:
            print('Fill Memory Bank [%d/%d]' %(i, len(loader)))