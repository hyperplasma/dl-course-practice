import torch
import torch.nn as nn
import torch.nn.functional as F

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))	# 将输入的特征图进行Global Average Pooling
        self.fc = nn.Sequential(
			nn.Linear(channel, channel // reduction, bias=False),
			nn.ReLU(inplace=True),
			nn.Linear(channel // reduction, channel, bias=False),
			nn.Sigmoid()
		)
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)	# (b, c, 1, 1)
        y = y.view(b, c)	# (b, c)
        y = self.fc(y)	# (b, c)
        y = y.view(b, c, 1, 1)	# (b, c, 1, 1)
        y = y.expand_as(x)	# (b, c, h, w)
        return x * y
    
class SEBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, reduction=16):
        super(SEBottleneck, self).__init__()
        mid_channels = in_channels // reduction
        
        # 第一层1x1卷积负责减少channels，降低计算量
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        
        # 第二层3x3卷积提取语义特征
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        
        # 第三层卷积将channel恢复到原始大小
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 插入SE模块
        self.se = SELayer(out_channels, reduction)
        
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        
        return residual + out

class SEResnet(nn.Module):
    def __init__(self, block, layers=[3, 4, 6, 3], num_classes=1000):
        super(SEResnet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self.make_group(block, 64, 64, layers[0])
        self.layer2 = self.make_group(block, 64, 128, layers[1], stride=2)
        self.layer3 = self.make_group(block, 128, 256, layers[2], stride=2)
        self.layer4 = self.make_group(block, 256, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x
    
    def make_group(self, block, in_channels, out_channels, blocks, stride=1, reduction=16):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        layers = []
        layers.append(block(in_channels, out_channels, stride, downsample, reduction))
        
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels, reduction=reduction))
            
        return nn.Sequential(*layers)

if __name__ == "__main__":
    x = torch.randn(2, 3, 112, 112)
    downsample = nn.Sequential(
        nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),
        nn.BatchNorm2d(128)
    )
    se = SEResnet(SEBottleneck, layers=[2, 3, 3, 3], num_classes=100)
    y = se(x)
    print(x.shape, y.shape)
