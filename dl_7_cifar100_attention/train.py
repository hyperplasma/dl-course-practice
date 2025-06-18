import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
from seresnet import SEResnet, SEBottleneck
import os
import time

# 训练函数
def train(model, train_loader, criterion, optimizer, device):
	model.train()
	running_loss = 0.0
	correct = 0
	total = 0

	for inputs, targets in train_loader:
		inputs, targets = inputs.to(device), targets.to(device)

		optimizer.zero_grad()
		outputs = model(inputs)
		loss = criterion(outputs, targets)
		loss.backward()
		optimizer.step()

		running_loss += loss.item() * inputs.size(0)
		_, predicted = outputs.max(1)
		total += targets.size(0)
		correct += predicted.eq(targets).sum().item()

	epoch_loss = running_loss / len(train_loader.dataset)
	epoch_acc = 100. * correct / total
	return epoch_loss, epoch_acc


if __name__ == "__main__":
	# 设置随机种子
	torch.manual_seed(42)

	# 设备配置
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	batch_size = 10
	lr = 0.01
	epochs = 10