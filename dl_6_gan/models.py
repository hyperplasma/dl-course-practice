import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super(Generator, self).__init__()
        
        self.backbone = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(),
			nn.Linear(128, 128),
			nn.ReLU(),
			nn.Linear(128, 128),
			nn.ReLU(),
			nn.Linear(128, 28 * 28),
   			# nn.Sigmoid()
            nn.Tanh()
		)
    
    def forward(self, z):
        img = self.backbone(z)	# (batch_size, 28 * 28)
        img = img.view(-1, 1, 28, 28)	# (batch_size, 1, 28, 28)
        return img

class Discriminator(nn.Module):
    def __init__(self, img_dim=28 * 28):
        super(Discriminator, self).__init__()

        self.backbone = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            # nn.Sigmoid()
            nn.Tanh()
        )
    
    def forward(self, img):
        img = img.view(img.size(0), -1)	# (batch_size, 1, img_dim)
        return self.backbone(img)


if __name__ == "__main__":
    data = torch.randn((10, 1, 28, 28))
    D = Discriminator()
    print(D(data))
    
    z = torch.randn((10, 100))
    G = Generator()
    print(G(z).shape)