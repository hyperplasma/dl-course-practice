from chrm_dataset import ChrmDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision as tv
import torch
from util import draw_progress_bar, plot_preds
import numpy as np
import matplotlib.pyplot as plt

class EMA():
    def __init__(self, model, decay=0.98):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()
        
    def register(self):
        """
        初始化时调用，将student的参数初始化至shadow
        """
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            self.shadow[name] = param.data.clone()
        
    def apply_shadow(self):
        """
        将student的参数备份到backup字典，然后将shadow的参数copy到student模型中
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]
        
    def restore(self):
        """
        将backup中的student参数copy回student模型中
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
            
    def update(self):
        """
        将student的参数通过EMA更新到shadow中
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
        

def train(semi_supervised=True, savefig_name="Figure.png", **kwargs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    transform = transforms.Compose([
        transforms.RandomRotation(90, fill=255),
        transforms.Resize((224, 224)),  
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomHorizontalFlip(p=0.5), # 随机水平翻转
        transforms.ToTensor()                  # 转换为张量
      
    ])
    
    # 根据模式选择数据集
    if semi_supervised:
        train_dataset = ChrmDataset(data_roots={'datasets/chrm_cls/with labels/train':1, 
                                        'datasets/chrm_cls/without labels':0},
                            transforms=transform)
    else:
        train_dataset = ChrmDataset(data_roots={'datasets/chrm_cls/with labels/train':1},
                            transforms=transform)


    batch_size = 32
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    test_dataset = ChrmDataset(data_roots={'datasets/chrm_cls/with labels/test':1},
                          transforms=transform)

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    student = tv.models.resnet18(pretrained=True)
    num_classes = 24
    student.fc = torch.nn.Linear(student.fc.in_features, num_classes)
    
    if semi_supervised:
        teacher = EMA(student, decay=0.98)
    
    n_epoch = 10
    ce_func = torch.nn.CrossEntropyLoss()
    mse_func = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(student.parameters(), lr=0.0001)
    step_per_epoch = len(train_dataset) // batch_size
    max_acc = 0
    
    mse_losses = []
    ce_losses = []
    for epoch in range(n_epoch):
        print("\nEpoch: #", epoch + 1)
        for step, (ims1, ims2, labels) in enumerate(train_dataloader):
            ims1, ims2, labels = ims1.to(device), ims2.to(device), labels.to(device)
            logit_s = student(ims1)
            
            if semi_supervised:
                # 半监督模式：使用教师模型和一致性正则化
                teacher.apply_shadow()
                logit_t = student(ims2)
                teacher.restore()
                
                # 计算一致性损失
                mse_loss = mse_func(torch.softmax(logit_s, dim=1), torch.softmax(logit_t, dim=1))
                mse_losses.append(mse_loss.item())
                
                # 计算有标签数据的交叉熵损失
                ce_loss = 0
                if (labels != -1).any():
                    keep_labels = torch.where(labels != -1)
                    ce_loss = ce_func(logit_s[keep_labels], labels[keep_labels])
                    ce_losses.append(ce_loss.item())
                else:
                    ce_losses.append(0.0)
                
                # 总损失
                total_loss = mse_loss + ce_loss * 2.0
            else:
                # 全监督模式：只使用交叉熵损失
                ce_loss = ce_func(logit_s, labels)
                ce_losses.append(ce_loss.item())
                total_loss = ce_loss
            
            # 反向传播和优化
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # 只在半监督模式下更新教师模型
            if semi_supervised:
                teacher.update()
                
            draw_progress_bar(step, step_per_epoch)
            
        # 执行在线验证
        print("\nOnline validation...")
        num_correct = 0
        num_total = 0
        student.eval()
        for xs_s, xs_t, labels in test_dataloader:
            xs_s, xs_t, labels = xs_s.to(device), xs_t.to(device), labels.to(device).long()
            with torch.no_grad():
                if semi_supervised:
                    # 半监督模式：使用教师模型进行预测
                    teacher.apply_shadow()
                    logit_t = student(xs_t)
                    teacher.restore()
                    
                    prob_t = torch.softmax(logit_t, dim=-1)
                    labels_t = torch.argmax(prob_t, dim=-1)
                else:
                    # 全监督模式：直接使用学生模型进行预测
                    logit_s = student(xs_s)
                    prob_s = torch.softmax(logit_s, dim=-1)
                    labels_t = torch.argmax(prob_s, dim=-1)
                
                num_correct += (labels_t == labels).sum().item()
                num_total += labels.shape[0]
        
        acc = num_correct / num_total
        ce_loss = np.mean(ce_losses)
        mse_loss = np.mean(mse_losses)
        print("Accuracy: {:.4f}, CE loss: {:.4f}, MSE loss: {:.4f}".format(acc, ce_loss, mse_loss))
        
        if acc > max_acc:
            max_acc = acc
            torch.save(student.state_dict(), 'checkpoints/chrm/student.pth')
            
    print("Max accuracy: {:.4f}".format(max_acc))
    
    # 数据可视化并保存
    plt.figure(figsize=(10, 6))
    plt.plot(ce_losses, label='CE Loss')
    if semi_supervised:
        plt.plot(mse_losses, label='MSE Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(savefig_name)
    plt.show()

    
if __name__ == '__main__':
    train(semi_supervised=True, savefig_name="Figure_semi.png")
    # train(semi_supervised=False, savefig_name="Figure_non_semi.png")