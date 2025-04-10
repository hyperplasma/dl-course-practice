import albumentations as A
from torch.utils.data import DataLoader
from skin_dataset import SkinDataset
import os

if __name__ == '__main__':
    # 实例化dataset和dataloader
    data_root = os.environ.get("ISIC_DATASET_ROOT")
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=180, p=0.5),
        A.GridDistortion(p=0.1),
        A.OpticalDistortion(p=0.1),
        A.Resize(height=128, width=128)
    ])
    batch_size = 2
    
    train_dataset = SkinDataset(data_root=data_root, subset="train", transform=transform)
    val_dataset = SkinDataset(data_root=data_root, subset="val", transform=transform)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(len(train_dataset), len(val_dataset))
    for i, (img, mask) in enumerate(train_dataloader):
        print(i, img.shape, mask.shape)
        break