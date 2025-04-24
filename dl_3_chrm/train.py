from chrm_dataset import ChrmDataset
from torch.utils.data import DataLoader
from torchvision import transforms as tv
import torch
from util import draw_progress_bar, plot_preds

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
        

if __name__ == '__main__':
    transforms = tv.Compose([
        tv.RandomRotation(90, fill=255),
        tv.Resize((224, 224)),  
        tv.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        tv.RandomHorizontalFlip(p=0.5), # 随机水平翻转
        tv.ToTensor()                  # 转换为张量
      
    ])
    
    train_dataset = ChrmDataset(data_roots={'/Users/hyperplasma/workspace/codes/Python/dl-test/dl_3_chrm/datasets/chrm_cls/with labels/train':1, 
                                      '/Users/hyperplasma/workspace/codes/Python/dl-test/dl_3_chrm/datasets/chrm_cls/without labels':0},
                          transforms=transforms)

    batch_size = 12
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    test_dataset = ChrmDataset(data_roots={'/Users/hyperplasma/workspace/codes/Python/dl-test/dl_3_chrm/datasets/chrm_cls/with labels/test':1},
                          transforms=transforms)

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    student = tv.models.resnet18(pretrained=True)
    student.fc = torch.nn.Linear(student.fc.in_features, 2)