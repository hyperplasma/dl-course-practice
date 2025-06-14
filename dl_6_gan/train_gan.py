import torch  
import torch.nn as nn  
import torch.optim as optim  
import torchvision  
import torchvision.transforms as transforms  
from torch.utils.data import DataLoader  
import matplotlib.pyplot as plt  
import numpy as np

class Generator(nn.Module):  
    def __init__(self, z_dim=100):  
        super(Generator, self).__init__()  
        self.main = nn.Sequential(  
            nn.Linear(z_dim, 256),  
            nn.ReLU(True),  
            nn.Linear(256, 512),  
            nn.ReLU(True),  
            nn.Linear(512, 1024),  
            nn.ReLU(True),  
            nn.Linear(1024, 28*28),  
            nn.Tanh() #[-1, 1]
        )  
  
    def forward(self, x):  
        img = self.main(x)  
        img = img.view(img.size(0), 1, 28, 28)  
        return img

class Discriminator(nn.Module):  
    def __init__(self):  
        super(Discriminator, self).__init__()  
        self.main = nn.Sequential(  
            nn.Linear(28*28, 512),  
            nn.LeakyReLU(0.2, inplace=True),  
            nn.Linear(512, 256),  
            nn.LeakyReLU(0.2, inplace=True),  
            nn.Linear(256, 1),  
            nn.Sigmoid()  
        )  
  
    def forward(self, x):  
        x = x.view(x.size(0), -1)  
        score = self.main(x)  
        return score

if __name__ == '__main__':
    # 参数设置  
    batch_size = 16  
    z_dim = 100  
    lr = 0.0002  
    epochs = 50  
    
    # 加载Fashion-MNIST数据集  
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])  
    mnist = torchvision.datasets.FashionMNIST(root='datasets/mnist', train=True, download=True, transform=transform)  
    dataloader = DataLoader(mnist, batch_size=batch_size, shuffle=True)  
    
    # 初始化模型  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    G = Generator(z_dim).to(device)  
    D = Discriminator().to(device)  
    
    # 损失函数和优化器  
    adversarial_loss = nn.BCELoss()  
    optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))  
    optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    
    # 准备真实和假的标签  
    real_labels = torch.ones(batch_size, 1, device=device)  
    fake_labels = torch.zeros(batch_size, 1, device=device)  
    #开始训练
    for epoch in range(epochs):  
        for i, (imgs, _) in enumerate(dataloader):              
            # 准备真实图像  
            imgs = imgs.to(device)  
            
            # 训练判别器  
            optimizer_D.zero_grad()  
            
            # 真实图像损失  
            real_score = D(imgs)
            real_loss = adversarial_loss(real_score, real_labels)  
            
            # 生成假图像  
            z = torch.randn(batch_size, z_dim, device=device)  
            fake_imgs = G(z)  
            
            # 假图像损失
            fake_score = D(fake_imgs.detach())    
            fake_loss = adversarial_loss(fake_score, fake_labels)  
            
            
            # 判别器总损失  
            d_loss = (real_loss + fake_loss) / 2  
            d_loss.backward()  
            optimizer_D.step()  
            
            # 训练生成器  
            optimizer_G.zero_grad()  
            
            # 生成假图像并计算损失  
            z = torch.randn(batch_size, z_dim, device=device)  
            fake_imgs = G(z)  
            g_loss = adversarial_loss(D(fake_imgs), real_labels)  
            
            # 生成器总损失  
            g_loss.backward()  
            optimizer_G.step()  
            
            # 打印训练进度  
            print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}] [D(x): {real_score.detach().mean().item():.4f}] [D(G(z)): {fake_score.mean().item():.4f}]")  
        
        # 每10个epoch保存一次生成的图像  
        if epoch % 5 == 0:  
            with torch.no_grad():  
                z = torch.randn(16, z_dim, device=device)  
                fake_imgs = G(z)  
                fake_imgs = fake_imgs / 2 + 0.5  # 反归一化  
                fake_imgs = fake_imgs.detach().cpu().numpy()  
                
                fig, axs = plt.subplots(4, 4, figsize=(8, 8))  
                cnt = 0  
                for i in range(4):  
                    for j in range(4):  
                        axs[i, j].imshow(fake_imgs[cnt, 0, :, :], cmap='gray')  
                        axs[i, j].axis('off')  
                        cnt += 1  
                plt.show()