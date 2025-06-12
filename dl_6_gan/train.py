import torchvision
from torch.utils.data import DataLoader
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from models import Generator, Discriminator

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

if __name__ == "__main__":
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,)),
	])
    
    mnist = torchvision.datasets.MNIST(
		root="datasets/mnist",
		train=True,
		download=False,
		transform=transform,
	)
    
    batch_size = 10
    dataloader = DataLoader(mnist, batch_size=batch_size, shuffle=True)
    
    # print(f"Number of batches: {len(dataloader)}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z_dim = 100
    G = Generator(z_dim=z_dim).to(device)
    D = Discriminator(img_dim=28 * 28).to(device)
    
    lr = 0.01
    optimizer_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    loss_func = torch.nn.BCELoss()
    
    g_losses = []
    d_losses = []
    
    num_epochs = 10
    step_per_epoch = len(mnist) // batch_size
    
    G.train()
    D.train()
    for epoch in range(num_epochs):
        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        train_g_losses = []
        train_d_losses = []
        for i, (real_imgs, _) in enumerate(dataloader):
            real_imgs = real_imgs.to(device)
            
            # 优化生成器
            z = torch.randn(batch_size, z_dim, device=device)
            fake_imgs = G(z)
            
            fake_labels = torch.zeros(batch_size, 1, device=device)
            real_labels = torch.ones(batch_size, 1, device=device)
            G_loss = loss_func(D(fake_imgs), real_labels)
            
            optimizer_D.zero_grad()
            G_loss.backward()
            optimizer_G.step()
            train_g_losses.append(G_loss.item())
            
            # 优化判别器
            D_loss = loss_func(D(real_imgs), real_labels) + loss_func(D(fake_imgs.detach()), fake_labels)
            
            optimizer_D.zero_grad()
            D_loss.backward()
            optimizer_D.step()
            train_d_losses.append(D_loss.item())
            
            draw_progress_bar(len(train_g_losses), step_per_epoch)
            
        g_loss_mean = np.mean(train_g_losses)
        d_loss_mean = np.mean(train_d_losses)
        print(f"Epoch [{epoch + 1}/{num_epochs}], G Loss: {g_loss_mean:.4f}, D Loss: {d_loss_mean:.4f}")
        g_losses.append(g_loss_mean)
        d_losses.append(d_loss_mean)
        
        if epoch % 5 == 0:
            with torch.no_grad():
                z = torch.randn(16, z_dim, device=device)
                fake_imgs = (G(z) / 2 + 0.5).detach().cpu().numpy()
                # fake_imgs = np.uint8(fake_imgs * 255)
                fig, axs = plt.subplots(4, 4, figsize=(8, 8))
                cnt = 0
                for i in range(4):
                    for j in range(4):
                        axs[i, j].imshow(fake_imgs[cnt, 0, :, :], cmap='gray')
                        axs[i, j].axis('off')
                        cnt += 1
                plt.show()
        