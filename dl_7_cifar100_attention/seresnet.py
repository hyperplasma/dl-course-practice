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

if __name__ == "__main__":
    x = torch.randn(2, 64, 112, 112)
    se = SELayer(64)
    y = se(x)
    print(x.shape, y.shape)

