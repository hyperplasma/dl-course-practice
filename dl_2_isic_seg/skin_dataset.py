from torch.utils.data import Dataset
import glob
import os.path as osp
import cv2
import os
import albumentations as A


class SkinDataset(Dataset):
    def __init__(self, data_root, subset="train", transform=None):
        self.transform = transform
        
        im_root = ""
        mask_root = ""
        if subset == "train":
            im_root = osp.join(data_root, "ISBI2016_ISIC_Part1_Training_Data")
            mask_root = osp.join(data_root, "ISBI2016_ISIC_Part1_Training_GroundTruth")
        else:
            im_root = osp.join(data_root, "ISBI2016_ISIC_Part1_Test_Data")
            mask_root = osp.join(data_root, "ISBI2016_ISIC_Part1_Test_GroundTruth")
        
        self.all_samples = []
        
        all_imfiles = glob.glob(osp.join(im_root, "*.jpg"))
        for imf in all_imfiles:
            maskf = osp.join(mask_root, osp.basename(imf).replace(".jpg", "_Segmentation.png"))
            if osp.exists(maskf):
                self.all_samples.append([imf, maskf])
        print("Total samples: ", len(self.all_samples))
        
    def __len__(self):
        return len(self.all_samples)
    
    def __getitem__(self, index):
        imf, maskf = self.all_samples[index]
        image = cv2.imread(imf, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(maskf, cv2.IMREAD_GRAYSCALE)
        if self.transform:
            trans = self.transform(image=image, mask=mask)
            image = trans["image"]
            mask = trans["mask"]
        
        # image = cv2.resize(image, (128, 128))
        # mask = cv2.resize(mask, (128, 128))
        
        mask[mask > 0] = 1  # 强制将前景标签设置为1
        image = image.transpose((2, 0, 1))
        return image, mask
    
    
if __name__ == "__main__":
    data_root = os.environ.get("ISIC_DATASET_ROOT")
    print("Data root: ", data_root)
    
    dataset = SkinDataset(data_root=data_root, subset="train")
    im, mask = dataset[10]
    print(im.shape, mask.shape)

