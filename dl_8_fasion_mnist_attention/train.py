import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import Generator, Discriminator
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# 1. 超参数
batch_size = 64
lr = 2e-4
epochs = 50
noise_dim = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_dir = "outputs/cgan_mnist"
os.makedirs(output_dir, exist_ok=True)

# 2. 数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
train_dataset = datasets.MNIST(root="datasets/mnist", train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# 3. 模型
G = Generator(noise_dim=noise_dim).to(device)
D = Discriminator().to(device)

# 4. 损失和优化器
adversarial_loss = nn.BCELoss()
optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

best_g_loss = float('inf')
best_epoch = 0

# 记录损失和准确率
g_loss_list = []
d_loss_list = []
d_acc_list = []

# 5. 训练
for epoch in range(1, epochs + 1):
    G.train()
    D.train()
    g_loss_epoch = 0
    d_loss_epoch = 0
    d_acc_epoch = 0
    batch_count = 0
    pbar = tqdm(train_loader, desc=f"Epoch [{epoch}/{epochs}]")
    for imgs, labels in pbar:
        batch_size_cur = imgs.size(0)
        real_imgs = imgs.to(device)
        labels = labels.to(device)
        valid = torch.ones(batch_size_cur, 1, device=device)
        fake = torch.zeros(batch_size_cur, 1, device=device)

        # 训练生成器
        optimizer_G.zero_grad()
        z = torch.randn(batch_size_cur, noise_dim, device=device)
        gen_labels = torch.randint(0, 10, (batch_size_cur,), device=device)
        gen_imgs = G(z, gen_labels)
        g_loss = adversarial_loss(D(gen_imgs, gen_labels), valid)
        g_loss.backward()
        optimizer_G.step()

        # 训练判别器
        optimizer_D.zero_grad()
        real_loss = adversarial_loss(D(real_imgs, labels), valid)
        fake_loss = adversarial_loss(D(gen_imgs.detach(), gen_labels), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        # 判别器准确率
        with torch.no_grad():
            real_pred = D(real_imgs, labels)
            fake_pred = D(gen_imgs.detach(), gen_labels)
            real_acc = (real_pred > 0.5).float().mean().item()
            fake_acc = (fake_pred < 0.5).float().mean().item()
            d_acc = (real_acc + fake_acc) / 2

        g_loss_epoch += g_loss.item()
        d_loss_epoch += d_loss.item()
        d_acc_epoch += d_acc
        batch_count += 1
        pbar.set_postfix(G_loss=g_loss.item(), D_loss=d_loss.item(), D_acc=d_acc)

    avg_g_loss = g_loss_epoch / batch_count
    avg_d_loss = d_loss_epoch / batch_count
    avg_d_acc = d_acc_epoch / batch_count
    g_loss_list.append(avg_g_loss)
    d_loss_list.append(avg_d_loss)
    d_acc_list.append(avg_d_acc)

    # 保留效果最好的模型和生成图片
    if avg_g_loss < best_g_loss:
        best_g_loss = avg_g_loss
        best_epoch = epoch
        # 保存模型
        torch.save(G.state_dict(), os.path.join(output_dir, f"best_generator.pth"))
        torch.save(D.state_dict(), os.path.join(output_dir, f"best_discriminator.pth"))
        # 保存生成图片
        G.eval()
        with torch.no_grad():
            z = torch.randn(10, noise_dim, device=device)
            labels_sample = torch.arange(0, 10, device=device)
            gen_imgs = G(z, labels_sample)
            gen_imgs = (gen_imgs + 1) / 2  # 反归一化到[0,1]
            from torchvision.utils import save_image
            save_image(gen_imgs, os.path.join(output_dir, f"best_sample.png"), nrow=10, normalize=False)

print(f"训练完成！已保存效果最好的模型（epoch={best_epoch}，G_loss={best_g_loss:.4f}）及生成图片。")

# 6. 绘制损失曲线
plt.figure()
plt.plot(range(1, epochs+1), g_loss_list, label='Generator Loss')
plt.plot(range(1, epochs+1), d_loss_list, label='Discriminator Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.grid(True)
loss_curve_path = os.path.join(output_dir, "loss_curve.png")
plt.savefig(loss_curve_path)
plt.close()
print(f"损失曲线已保存到 {loss_curve_path}")

# 7. 绘制判别器准确率曲线
plt.figure()
plt.plot(range(1, epochs+1), d_acc_list, label='Discriminator Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Discriminator Accuracy Curve')
plt.legend()
plt.grid(True)
acc_curve_path = os.path.join(output_dir, "accuracy_curve.png")
plt.savefig(acc_curve_path)
plt.close()
print(f"判别器准确率曲线已保存到 {acc_curve_path}")