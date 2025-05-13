import torchvision.models.segmentation as seg
import torch

def get_fcn_resnet50(num_classes=2, pretrained=False):
    model = seg.fcn_resnet50(weights=None if not pretrained else 'segmentation')
    # 修改模型输出层
    model.classifier[-1] = torch.nn.Conv2d(model.classifier[-1].in_channels, num_classes, kernel_size=(1, 1))
    model.aux_classifier = None
    return model

# 获取另一种模型deeplabv3 resnet101
def get_deeplabv3_resnet101(num_classes=2, pretrained=False):
    model = seg.deeplabv3_resnet101(weights=None if not pretrained else 'segmentation')
    # 修改模型输出层
    model.classifier[-1] = torch.nn.Conv2d(model.classifier[-1].in_channels, num_classes, kernel_size=(1, 1))
    return model

if __name__ == '__main__':
    model = get_deeplabv3_resnet101(num_classes=2, pretrained=False)
    model.eval()
    # 输入(3, 224, 224)，输出(10, 224, 224)
    im = torch.randn((1, 3, 224, 224))
    pred = model(im)["out"]
    print(pred.shape)
    