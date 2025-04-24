from torch.utils.data import Dataset
import glob
import os.path as osp
from PIL import Image
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class ChrmDataset(Dataset):
    def __init__(self, data_roots, transforms=None):
        '''
        data_roots: 数据集路径字典 e.g.,{'labeled path':1, 'unlabeled path':0}，其中0和1表示无标签和有标签
        '''
        self.transforms = transforms
        self.samples = []
        for data_root in data_roots:
            has_label = data_roots[data_root]
            if has_label:
                for k in range(1, 25):
                    k_files = glob.glob(osp.join(data_root, str(k), '*.PNG'))
                    for f in k_files:
                        self.samples.append([f, k-1])#标签从0开始
            else:
                files = glob.glob(osp.join(data_root, '*.PNG'))
                for f in files:
                    self.samples.append([f, -1])#-1表示无标签
                    

        np.random.shuffle(self.samples)

        print('Found images:', len(self.samples))
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        im_f, label = self.samples[index]
        pil_im = Image.open(im_f).convert('RGB')

        if self.transforms:
            im1 = self.transforms(pil_im)
            im2 = self.transforms(pil_im)
        else:
            im1 = im_f
            im2 = im_f           
       
        return im1, im2, label

    

if __name__ == '__main__':
    from torchvision import transforms as tv
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

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

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2)

    for im1, im2, label in dataloader:
        print(im1.shape, im2.shape, label)
        im1 = im1[0, ...].numpy().transpose((1,2,0))
        im2= im2[0,...].numpy().transpose((1,2,0))
        plt.subplot(1,2,1)
        plt.imshow(im1, cmap='gray')
        plt.subplot(1,2,2)
        plt.imshow(im2, cmap='gray')
        plt.show()
