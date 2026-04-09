import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
from thop import profile
from mobileResnet18 import MobileResNet
from torch.utils.data import random_split
from torchvision import datasets
import matplotlib.pyplot as plt  # 导入matplotlib库

# 固定随机种子
import random
import numpy as np

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 定义训练函数
def train(model, train_loader, criterion, optimizer, device, train_losses):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计损失和准确率
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    # 计算每轮时间
    epoch_time = time.time() - start_time

    # 计算平均损失和准确率
    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total

    # 计算FLOPs和参数量
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    flops, params = profile(model, inputs=(dummy_input,))

    print(f"训练损失: {avg_loss:.4f}, 训练准确率: {accuracy:.2f}%, 每轮时间: {epoch_time:.2f}s, 参数量: {params / 1e6:.2f}M, FLOPs: {flops / 1e9:.2f}G")

    # 记录训练损失
    train_losses.append(avg_loss)

# 定义验证函数
def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 统计损失和准确率
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    # 计算平均损失和准确率
    avg_loss = val_loss / len(val_loader)
    accuracy = 100. * correct / total

    print(f"验证损失: {avg_loss:.4f}, 验证准确率: {accuracy:.2f}%")
    return avg_loss, accuracy

# 主函数
if __name__ == '__main__':
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # 加载训练集
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = datasets.cifar.CIFAR10(root=r'C:\\Users\\Administrator\Desktop\\msnet\\data\\image',download=True,
                                      transform=train_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 加载验证集
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset, val_dataset = random_split(train_dataset, [round(0.8 * len(train_dataset)), round(0.2 * len(train_dataset))])
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 初始化模型
model = MobileResNet(num_classes=10).to(device)  # 将num_classes改为10，因为CIFAR-10有10个类别
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
num_epochs = 100
best_val_accuracy = 0.0  # 用于保存最佳验证准确率
train_losses = []  # 用于保存每轮训练损失
val_losses = []  # 用于保存每轮验证损失

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    train(model, train_loader, criterion, optimizer, device, train_losses)

        # 每五轮验证一次
    if (epoch + 1) % 5 == 0:
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)  # 记录验证损失

        # 如果当前验证准确率优于之前的最佳准确率，则保存模型
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), r'C:\\Users\\Administrator\Desktop\\msnet\\results\\best_model.pth')
            print(f"模型已保存，验证准确率: {best_val_accuracy:.2f}%")
            torch.autograd.set_detect_anomaly(True)
