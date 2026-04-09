import torch
import torch.nn as nn
import torch.nn.functional as F

# Squeeze-and-Excitation模块
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1)

    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, 1)
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w

# h-swish激活函数
class HSwish(nn.Module):
    def forward(self, x):
        return x * F.relu6(x + 3) / 6

# Bottleneck结构
class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, exp_channels, se, nl):
        super(Bottleneck, self).__init__()
        self.use_res_connect = stride == 1 and in_channels == out_channels
        activation = nn.ReLU if nl == 'RE' else HSwish

        self.conv = nn.Sequential(
            # 1x1 pw
            nn.Conv2d(in_channels, exp_channels, 1, bias=False),
            nn.BatchNorm2d(exp_channels),
            activation(),
            # 3x3 dw
            nn.Conv2d(exp_channels, exp_channels, kernel_size, stride, kernel_size // 2, groups=exp_channels, bias=False),
            nn.BatchNorm2d(exp_channels),
            activation(),
            # SE
            SEBlock(exp_channels) if se else nn.Identity(),
            # 1x1 pw-linear
            nn.Conv2d(exp_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

# MobileNetV3-Large主干
class MobileNetV3(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV3, self).__init__()
        layers = []
        activation = HSwish

        # Stem
        layers.append(nn.Conv2d(3, 16, 3, 2, 1, bias=False))
        layers.append(nn.BatchNorm2d(16))
        layers.append(activation())

        # 配置表: in, out, k, s, exp, se, nl
        cfgs = [
            # k, exp, out, se, nl, s
            [3, 16, 16, False, 'RE', 1],
            [3, 64, 24, False, 'RE', 2],
            [3, 72, 24, False, 'RE', 1],
            [5, 72, 40, True,  'RE', 2],
            [5, 120, 40, True,  'RE', 1],
            [5, 120, 40, True,  'RE', 1],
            [3, 240, 80, False, 'HS', 2],
            [3, 200, 80, False, 'HS', 1],
            [3, 184, 80, False, 'HS', 1],
            [3, 184, 80, False, 'HS', 1],
            [3, 480, 112, True, 'HS', 1],
            [3, 672, 112, True, 'HS', 1],
            [5, 672, 160, True, 'HS', 2],
            [5, 960, 160, True, 'HS', 1],
            [5, 960, 160, True, 'HS', 1],
        ]

        in_channels = 16
        for k, exp, out, se, nl, s in cfgs:
            layers.append(Bottleneck(in_channels, out, k, s, exp, se, nl))
            in_channels = out

        # Head
        layers.append(nn.Conv2d(in_channels, 960, 1, bias=False))
        layers.append(nn.BatchNorm2d(960))
        layers.append(activation())
        layers.append(nn.AdaptiveAvgPool2d(1))
        layers.append(nn.Conv2d(960, 1280, 1))
        layers.append(activation())
        layers.append(nn.Flatten())
        layers.append(nn.Linear(1280, num_classes))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# 测试
if __name__ == "__main__":
    net = MobileNetV3(num_classes=5)
    x = torch.randn(1, 3, 224, 224)
    y = net(x)
    print(y.shape)  # 应输出 torch.Size([1, 1000])


