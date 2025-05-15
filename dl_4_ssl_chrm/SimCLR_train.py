import torch
import torch.nn as nn
from torchvision.models import resnet18
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from chrm_dataset import ChrmDataset
import sys
import matplotlib.pyplot as plt

def draw_progress_bar(cur, total, bar_len=50):
    """
        Print progress bar during training
    """
    cur_len = int(cur / total * bar_len)
    sys.stdout.write('\r')
    sys.stdout.write("[{:<{}}] {}/{}".format("=" * cur_len, bar_len, cur, total))
    sys.stdout.flush()

def plot_preds(ims, preds, masks):
    '''
    用于可视化训练中间结果
    '''
    preds = torch.softmax(preds, dim=1)

    ims = ims.detach().cpu().numpy()  
    preds = preds.detach().cpu().numpy()  
    masks = masks.detach().cpu().numpy()  
   
    plt.subplot(1, 3, 1)   
    plt.imshow(np.uint8(ims[0, ...]).transpose((1,2,0)))  

    plt.subplot(1, 3, 2) 
    plt.imshow(preds[0,1, ...], cmap='gray')  
    
    plt.subplot(1, 3, 3) 
    plt.imshow(masks[0, ...], cmap='gray')  
 
    plt.show()

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
                           transforms=transform)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# print('len(train_loader.dataset):', len(train_loader.dataset))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimCLR(z_dim=256).to(device)
# print(model)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 10
step_per_epoch = len(train_dataset) // batch_size
train_losses = []
for epoch in range(num_epochs):
    print(f"Epoch [{epoch + 1} / {num_epochs}]")
    model.train()
    for step, (img1, img2, _) in enumerate(train_loader):
        img1, img2 = img1.to(device), img2.to(device)
        # print('img1.shape:', img1.shape, 'img2.shape:', img2.shape)
        
        z1, t = model(img1)
        z2, t = model(img2)
        
        z1_norm = z1 / z1.norm(dim=1, keepdim=True)
        z2_norm = z2 / z2.norm(dim=1, keepdim=True)
        similar_matrix = t * (z1_norm @ z2_norm.t())
        # print('similar_matrix.shape:', similar_matrix.shape)
        
        labels = torch.arange(similar_matrix.shape[-1]).to(device)
        loss = criterion(similar_matrix, labels)
        # print('loss:', loss.item())
        train_losses.append(loss.item())
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        draw_progress_bar(step, step_per_epoch)
    
    print(f"\nTrain Loss: {np.mean(train_losses):.4f}")