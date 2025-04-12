import albumentations as A
from torch.utils.data import DataLoader
from skin_dataset import SkinDataset
import os
import torch
import numpy as np
import pandas as pd
from util import draw_progress_bar, plot_preds
from dice_loss import MultiClassDiceCoeff, MultiClassDiceLoss
import time
from datetime import datetime
import matplotlib.pyplot as plt
from models import get_fcn_resnet50

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
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.95)

    # 开辟保存目录，存储模型参数和训练参数
    save_path = './checkpoints/resnet50'
    os.makedirs(save_path, exist_ok=True)

    df_summary = pd.DataFrame(columns=['time', 'step', 'train dice loss', 'eval dsc'])
    df_summary.to_csv(os.path.join(save_path, "training_summary.csv"), index=False)

    # 定义损失函数
    ce_loss_func = torch.nn.CrossEntropyLoss()

    dc_coeff_func = MultiClassDiceCoeff(num_classes=2)
    dc_loss_func = MultiClassDiceLoss(num_classes=2)

    # 开始迭代训练
    model.train()
    num_epochs = 100
    max_dsc = 0.0
    step_per_epoch = len(train_dataset) // batch_size
    for k in range(num_epochs):
        print("Epoch: #", k + 1)
        t1 = time.time()
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

        t2 = time.time()

        # 在线验证
        model.eval()
        eval_dsc = 0.0
        show_yet = False
        with torch.no_grad():
            for ims, masks in val_dataloader:
                ims = ims.to(device).float()
                masks = masks.to(device).long()

                preds = model(ims)["out"]

                dsc = dc_coeff_func(preds, masks)
                eval_dsc += dsc.item()

                # 如果epoch是5的整数倍，且没有显示过分割结果，则可视化展示
                # if not show_yet and k % 5 == 0:
                #     plot_preds(ims, preds, masks)
                #     show_yet = True

        train_dice_loss = np.mean(ce_losses) + np.mean(dc_losses)
        eval_dsc = eval_dsc / len(val_dataset)
        print("\nTrain Dice Loss:", train_dice_loss, ", Eval Dice:", eval_dsc, " Train time:", t2 - t1)

        # 保存最佳模型参数
        checkpoint_file = os.path.join(save_path, "best_weights.pth")
        if max_dsc <= eval_dsc:
            max_dsc = eval_dsc
            print('Saving weights to %s' % (checkpoint_file))
            torch.save(model.state_dict(), checkpoint_file)

        # 保存训练参数
        current_time = "%s" % datetime.now()  # 获取当前时间
        step = "Step[%d]" % k
        str_train_loss = "%f" % train_dice_loss
        str_eval_dsc = "%f" % eval_dsc

        list_info = [current_time, step, str_train_loss, str_eval_dsc]
        df_summary = pd.DataFrame([list_info])
        df_summary.to_csv(os.path.join(save_path, "training_summary.csv"), mode='a', header=False, index=False)
    