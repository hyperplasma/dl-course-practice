import pandas as pd
import matplotlib.pyplot as plt
import os

# 读取CSV
csv_path = os.path.join('outputs', 'fashion_mnist', 'accuracy.csv')
df = pd.read_csv(csv_path)

# 绘制折线图
plt.figure(figsize=(8, 5))
plt.plot(df['epoch'], df['train_acc'], label='Train Accuracy', marker='o')
plt.plot(df['epoch'], df['test_acc'], label='Test Accuracy', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Fashion MNIST Accuracy Curve')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()