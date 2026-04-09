import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image
from mobileResnet18 import MobileResNet
from mobilenet import MobileNetV3
import time
from thop import profile  # 用于计算 FLOPs 和参数量
from collections import OrderedDict
# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 图像预处理函数，避免重复定义
normalizes= transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
def get_image_transform():
    return transforms.Compose([
        transforms.Resize((224,224)),  # (256, 256) 区别
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def predict_image(model, image_path):
    try:
        model.eval()

        # 图像预处理
        transform = get_image_transform()

        # 加载并处理图像
        image = Image.open(image_path).convert('RGB')  # 确保图像为RGB格式
        image = transform(image).unsqueeze(0)

        # 预测
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)

        # 类别映射
        classes = ['Non-destructive coal', 'Destructive coal', 'Strongly destructive coal',
                   'Pulverized coal', 'Fully pulverized coal']

        return classes[predicted.item()]
    except Exception as e:
        print(f"预测过程中发生错误: {e}")
        return None
def evaluate_model(test_loader, model):
    model.eval()
    total_loss = 0.0
    correct = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / len(test_loader.dataset)
    return avg_loss, accuracy

def calculate_flops_and_params(model):
    try:
        input_tensor = torch.randn(1, 3, 224, 224)  # 输入张量
        flops, params = profile(model, inputs=(input_tensor,))
        return flops, params
    except Exception as e:
        print(f"计算 FLOPs 和参数量时发生错误: {e}")
        return None, None



def calculate_throughput(test_loader, model):
    try:
        model.eval()
        start_time = time.time()
        for images, _ in test_loader:
            with torch.no_grad():
                _ = model(images)
        end_time = time.time()
        total_images = len(test_loader.dataset)
        throughput = total_images / (end_time - start_time)
        return throughput
    except Exception as e:
        print(f"计算吞吐量时发生错误: {e}")
        return None

if __name__ == '__main__':
    try:
        # 加载测试数据集
        test_transform = get_image_transform()
        test_dataset = datasets.CIFAR10(root=r'C:\Users\Administrator\Desktop\msnet\data\image', train=False, transform=test_transform, download=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # 加载模型
        model = MobileResNet()
        model.load_state_dict(torch.load(r'C:\\Users\\Administrator\Desktop\\msnet\\results\\best_model.pth', weights_only=True),
                              strict=False)

        # 计算损失和准确率
        start_time = time.time()
        avg_loss, accuracy = evaluate_model(test_loader, model)
        end_time = time.time()

        # 计算 FLOPs 和参数量
        flops, params = calculate_flops_and_params(model)

        # 计算吞吐量
        throughput = calculate_throughput(test_loader, model)

        # 输出结果
        if flops is not None and params is not None and throughput is not None:
            print(f'测试集平均损失: {avg_loss:.4f}')
            print(f'测试集准确率: {accuracy :.2f}%')
            print(f'每轮时间: {end_time - start_time:.2f} 秒')
            print(f'模型参数量: {params / 1e6:.2f} M')
            print(f'模型 FLOPs: {flops / 1e9:.2f} G')
            print(f'吞吐量: {throughput:.2f} images/second')
    except Exception as e:
        print(f"主程序运行过程中发生错误: {e}")
