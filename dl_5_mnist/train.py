from torchvision import datasets, transforms
import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys
import os
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

class DeepImageEncoder(nn.Module):
    def __init__(self, output_dim=512):
        super(DeepImageEncoder, self).__init__()
        
        # 使用多个 Conv-BN-ReLU 模块构建更深的网络
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),

            nn.AdaptiveAvgPool2d((1, 1))  # 自适应池化到固定输出尺寸 (1x1)
        )

        self.fc = nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.features(x)       # 输出形状为 [B, 128, 1, 1]
        x = x.view(x.size(0), -1)  # 展平 -> [B, 128]
        x = self.fc(x)             # 映射到输出维度 [B, 512]
        return x

def contrastive_loss(image_embeddings, text_embeddings, temperature=0.07):
    image_embeddings = image_embeddings/image_embeddings.norm(dim=1, keepdim=True)
    text_embeddings = text_embeddings/text_embeddings.norm(dim=1, keepdim=True)
    similar_matrix = (image_embeddings@text_embeddings.t()/temperature)#(batch_size, batch_size)
    
    labels = torch.arange(similar_matrix.shape[0])
    loss_i = F.cross_entropy(similar_matrix, labels)
    loss_t = F.cross_entropy(similar_matrix.t(), labels)
    return (loss_i + loss_t) / 2

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    data_root = './datasets/mnist'
    train_dataset = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)

    batch_size = 20
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # print(f'Training samples: {len(train_dataset)}')
    # print(f'Test samples: {len(test_dataset)}')
    
    # 实例化图像编码器
    image_encoder = DeepImageEncoder(output_dim=512)
    # 实例化文本编码器
    bert_base_dir = 'models/bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(bert_base_dir)
    text_encoder = BertModel.from_pretrained(bert_base_dir) 
 
    optimizer = optim.Adam(list(image_encoder.parameters()) + list(text_encoder.parameters()), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # 创建保存可视化结果的目录
    save_dir = os.path.join('output', 'visualization')
    os.makedirs(save_dir, exist_ok=True)
    
    # 用于记录训练过程的指标
    epoch_losses = []
    
    num_epochs = 20
    step_per_epoch = len(train_dataset) // batch_size
    image_encoder.train()
    text_encoder.train()
    
    for epoch in range(num_epochs):
        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        train_losses = []
        for imgs, labels in train_loader:
            # 图像编码器前向传播
            image_features = image_encoder(imgs)  # (20, 768)
   
            text_descriptions = ["A picture of number {}".format(i) for i in labels]
            text_input = tokenizer(text_descriptions, return_tensors="pt", padding=True, truncation=True)
   
            text_embeddings = text_encoder(**text_input).last_hidden_state.mean(dim=1)
            loss = contrastive_loss(image_features, text_embeddings)
            train_losses.append(loss.item())
   
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
   
            draw_progress_bar(len(train_losses), step_per_epoch)
        
        # 计算并记录当前epoch的平均损失
        avg_loss = sum(train_losses) / len(train_losses)
        epoch_losses.append(avg_loss)
        print(f'\nTrain Loss: {avg_loss:.4f}')
        
        # 每个epoch结束后保存当前的损失曲线
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, 'b-', label='Training Loss')
        plt.title('Training Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # 保存损失曲线图
        plt.savefig(os.path.join(save_dir, 'training_loss.png'))
        plt.close()
        
        # 如果需要，可以保存最新的模型检查点
        if (epoch + 1) % 5 == 0:  # 每5个epoch保存一次模型
            model_save_dir = os.path.join('output', 'checkpoints')
            os.makedirs(model_save_dir, exist_ok=True)
            
            torch.save({
                'epoch': epoch + 1,
                'image_encoder_state_dict': image_encoder.state_dict(),
                'text_encoder_state_dict': text_encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, os.path.join(model_save_dir, f'checkpoint_epoch_{epoch+1}.pt'))
    
    print(f"\n训练完成！可视化结果已保存至: {save_dir}")