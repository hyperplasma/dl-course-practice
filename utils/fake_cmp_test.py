import matplotlib.pyplot as plt
import numpy as np

# 设置图片清晰度和中文显示
plt.rcParams['figure.dpi'] = 300
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False

# 生成模拟训练损失数据
epochs = np.arange(1, 21)  # 20个训练周期
resnet_loss = np.exp(-0.1 * epochs) + np.random.normal(0, 0.05, size=epochs.shape)
se_resnet_loss = np.exp(-0.12 * epochs) + np.random.normal(0, 0.05, size=epochs.shape)

# 创建图形
plt.figure(figsize=(10, 6))

# 绘制损失对比曲线
plt.plot(epochs, resnet_loss, 'o-', label='ResNet', linewidth=2)
plt.plot(epochs, se_resnet_loss, 's-', label='SE-ResNet', linewidth=2)

# 添加标题、标签和图例
plt.title('ResNet与SE-ResNet在CIFAR100上的训练损失对比', fontsize=15)
plt.xlabel('训练轮次 (Epoch)', fontsize=12)
plt.ylabel('交叉熵损失 (Cross Entropy Loss)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10)
plt.xticks(epochs[::2])  # 每两个epoch显示一个刻度

plt.tight_layout()
plt.show()    