import torch
import torch.nn as nn
from torchvision.models import resnet18
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from chrm_dataset import ChrmDataset

class SimCLR(nn.Module):
    def __init__(self, base_model=resnet18, z_dim=128):
        super(SimCLR, self).__init__()
        # 定义encoder
        self.encoder = base_model(pretrained=False)
        # 将fc层替换为projector
        self.encoder.fc = nn.Linear(self.encoder.fc.in_features, z_dim)
        
        # 定义可学习的温控系数
        self.t = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
    def forward(self, x):
        z = self.encoder(x)
        
        return z, self.t.exp()


transform = transforms.Compose([
    transforms.RandomRotation(90, fill=225),
    transforms.Resize((64, 64)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
])

train_dataset = ChrmDataset(data_roots={'datasets/chrm_cls/with labels/train':1, 
                                      'datasets/chrm_cls/without labels':0},
                           transforms=transforms)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

print('len(train_loader.dataset):', len(train_loader.dataset))