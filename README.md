# MSNet - Depthwise Separable + Channel Attention

本项目实现一种**轻量化卷积模块**，将深度可分离卷积与通道加权注意力（SE/Channel Attention）结合，用于构建轻量级分类模型。当前代码包含改进的 MobileResNet、MobileNetV3 与 GhostNet Large，并提供训练与评估脚本。

## 核心思路
在 `src/mobileResnet18.py` 中实现了 `DepthwiseSeparableSEConv`：

1. **Depthwise Conv**：按通道进行深度卷积（groups=in_channels）
2. **Channel Attention（SE）**：全局平均池化 → 两层全连接 → Sigmoid 生成通道权重
3. **Pointwise Conv**：1×1 逐点卷积融合通道
4. **BN + ReLU**：稳定训练并提升非线性表达

该设计在保证模型表达力的同时，显著降低参数量和计算量，适合资源受限场景。

## 代码结构
- `src/mobileResnet18.py`：深度可分离 + 通道注意力的 MobileResNet（核心模块在此）
- `src/mobilenet.py`：MobileNetV3-Large（带 SE）
- `src/ghostnet.py`：GhostNet Large（带 SE）
- `src/train.py`：ImageFolder 分类训练（默认 MobileNetV3）
- `src/traincifar10.py`：CIFAR-10 训练（默认 MobileResNet）
- `src/predict.py`：加载模型并在测试集评估
- `data/`：数据集目录（示例路径）
- `results/`：训练输出与模型权重

## 环境依赖
- Python 3.x
- torch
- torchvision
- thop
- pillow
- matplotlib

安装示例：
```bash
pip install torch torchvision thop pillow matplotlib
```

## 使用方法
### 1. 自定义数据集（ImageFolder）训练
`src/train.py` 使用 `datasets.ImageFolder`，请根据本地数据路径修改：

```python
train_dataset = datasets.ImageFolder(root='你的训练集路径', transform=train_transform)
```

运行：
```bash
python src/train.py
```

### 2. CIFAR-10 训练
`src/traincifar10.py` 会自动下载 CIFAR-10 到 `data/image`：

```bash
python src/traincifar10.py
```

### 3. 模型评估 / 推理
`src/predict.py` 会加载 `results/best_model.pth` 并在测试集评估。请确认：
- 模型权重路径正确
- 类别数量与模型一致

运行：
```bash
python src/predict.py
```

## 注意事项
- 训练/推理脚本中含有**Windows 绝对路径**示例，请改为你自己的路径。
- `MobileResNet` 默认 `num_classes=5`，CIFAR-10 训练脚本中已改为 10 类。

## License
MIT License. See `LICENSE`.
