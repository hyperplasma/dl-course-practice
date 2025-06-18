import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, noise_dim=100, label_dim=10, img_shape=(1, 28, 28)):
        super().__init__()
        self.label_emb = nn.Embedding(label_dim, label_dim)
        input_dim = noise_dim + label_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, int(torch.prod(torch.tensor(img_shape)))),
            nn.Tanh()
        )
        self.img_shape = img_shape

    def forward(self, noise, labels):
        # labels: [B], noise: [B, noise_dim]
        label_input = self.label_emb(labels)
        x = torch.cat((noise, label_input), dim=1)
        img = self.model(x)
        img = img.view(img.size(0), *self.img_shape)
        return img

class Discriminator(nn.Module):
    def __init__(self, label_dim=10, img_shape=(1, 28, 28)):
        super().__init__()
        self.label_emb = nn.Embedding(label_dim, label_dim)
        input_dim = label_dim + int(torch.prod(torch.tensor(img_shape)))
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        # img: [B, 1, 28, 28], labels: [B]
        img_flat = img.view(img.size(0), -1)
        label_input = self.label_emb(labels)
        x = torch.cat((img_flat, label_input), dim=1)
        validity = self.model(x)
        return validity

if __name__ == "__main__":
    # 简单测试
    noise = torch.randn(8, 100)
    labels = torch.randint(0, 10, (8,))
    G = Generator()
    D = Discriminator()
    fake_imgs = G(noise, labels)
    out = D(fake_imgs, labels)
    print("生成图片 shape:", fake_imgs.shape)
    print("判别器输出 shape:", out.shape)