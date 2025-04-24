from chrm_dataset import ChrmDataset
from torchvision import transforms as tv

if __name__ == '__main__':
    transforms = tv.Compose([
        tv.RandomRotation(90, fill=255),
        tv.Resize((224, 224)),  
        tv.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        tv.RandomHorizontalFlip(p=0.5), # 随机水平翻转
        tv.ToTensor()                  # 转换为张量
      
    ])
    
    dataset = ChrmDataset(data_roots={'/Users/hyperplasma/workspace/codes/Python/dl-test/dl_3_chrm/datasets/chrm_cls/with labels/train':1, 
                                      '/Users/hyperplasma/workspace/codes/Python/dl-test/dl_3_chrm/datasets/chrm_cls/without labels':0},
                          transforms=transforms)
