import torch
import torch.nn as nn
from torchvision.models import resnet18
import numpy as np

class SimCLR(nn.Module):
    def __init__(self, base_model=resnet18, z_dim=128):
        super(SimCLR, self).__init__()
        # 定义encoder
        self.encoder = base_model(pretrained=True)
        # 将fc层替换为projector
        self.encoder.fc = nn.Linear(self.encoder.fc.in_features, z_dim)
        
        # 定义可学习的温控系数
        self.t = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
    def forward(self, x):
        z = self.encoder(x)
        
        return z, self.t.exp()