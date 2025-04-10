import albumentations as A
from torch.utils.data import DataLoader
from skin_dataset import SkinDataset
import os
from models import get_fcn_resnet50
import torch
import numpy as np
from util import draw_progress_bar
from dice_loss import MultiClassDiceCoeff, MultiClassDiceLoss

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
    
    # 实例化模型
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = get_fcn_resnet50(num_classes=2).to(device)
    
    # 定义超参数
    num_epochs = 100
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.95)
    
    ce_loss_func = torch.nn.CrossEntropyLoss()
    
    dc_coeff_func = MultiClassDiceCoeff(num_classes=2)
    dc_loss_func = MultiClassDiceLoss(num_classes=2)
    
    # 开始迭代训练
    model.train() 
    step_per_epoch = len(train_dataset) // batch_size
    for k in range(num_epochs):
        print("Epoch: #", k + 1)
        ce_losses = []
        dc_losses = []
        for step, (ims, masks) in enumerate(train_dataloader):
            ims = ims.to(device).float()
            masks = masks.to(device).long()
            
            preds = model(ims)["out"]
            
            ce_loss = ce_loss_func(preds, masks)
            dc_loss = dc_loss_func(preds, masks)
            
            ce_losses.append(ce_loss.item())
            dc_losses.append(dc_loss.item())
            total_loss = 2 * ce_loss + dc_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            draw_progress_bar(step, step_per_epoch)
        
        # 在线验证
        model.eval()
        eval_dsc = 0.0
        with torch.no_grad():
            for ims, masks in val_dataloader:
                ims = ims.to(device).float()
                masks = masks.to(device).long()

                preds = model(ims)["out"]

                dsc = dc_coeff_func(preds, masks)
                eval_dsc += dsc.item()
        
        print("\nCE Loss:", np.mean(ce_losses), ", Dice Loss:", np.mean(dc_losses), ", Eval Dice:", eval_dsc / len(val_dataset))
            