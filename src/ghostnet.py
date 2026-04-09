import torch
import torch.nn as nn
import torch.nn.functional as F

# GhostModule
class GhostModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        init_channels = int(out_channels / ratio)
        new_channels = out_channels - init_channels

        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential()
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential()
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.primary_conv[0].out_channels + self.cheap_operation[0].out_channels, :, :]

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

# GhostBottleneck
class GhostBottleneck(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, dw_kernel_size, stride, use_se):
        super(GhostBottleneck, self).__init__()
        self.stride = stride
        self.use_se = use_se

        self.ghost1 = GhostModule(in_channels, mid_channels, relu=True)
        if stride > 1:
            self.dw_conv = nn.Conv2d(mid_channels, mid_channels, dw_kernel_size, stride, dw_kernel_size // 2, groups=mid_channels, bias=False)
            self.dw_bn = nn.BatchNorm2d(mid_channels)
        else:
            self.dw_conv = None

        self.se = SEBlock(mid_channels) if use_se else nn.Sequential()
        self.ghost2 = GhostModule(mid_channels, out_channels, relu=False)

        if stride == 1 and in_channels == out_channels:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, dw_kernel_size, stride, dw_kernel_size // 2, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.ghost1(x)
        if self.dw_conv is not None:
            out = self.dw_conv(out)
            out = self.dw_bn(out)
        out = self.se(out)
        out = self.ghost2(out)
        out += self.shortcut(x)
        return out

# GhostNet Large
class GhostNetLarge(nn.Module):
    def __init__(self, num_classes=5):
        super(GhostNetLarge, self).__init__()
        cfgs = [
            # k, exp, c, SE, s
            [3,  16,  16, 0, 1],
            [3,  48,  24, 0, 2],
            [3,  72,  24, 0, 1],
            [5,  72,  40, 1, 2],
            [5, 120,  40, 1, 1],
            [3, 240,  80, 0, 2],
            [3, 200,  80, 0, 1],
            [3, 184,  80, 0, 1],
            [3, 184,  80, 0, 1],
            [3, 480, 112, 1, 1],
            [3, 672, 112, 1, 1],
            [5, 672, 160, 1, 2],
            [5, 960, 160, 0, 1],
            [5, 960, 160, 1, 1],
        ]
        layers = []
        input_channel = 16
        layers.append(nn.Conv2d(3, input_channel, 3, 2, 1, bias=False))
        layers.append(nn.BatchNorm2d(input_channel))
        layers.append(nn.ReLU(inplace=True))

        for k, exp, c, se, s in cfgs:
            layers.append(GhostBottleneck(input_channel, exp, c, k, s, se))
            input_channel = c

        layers.append(nn.Conv2d(input_channel, 960, 1, 1, 0, bias=False))
        layers.append(nn.BatchNorm2d(960))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.AdaptiveAvgPool2d(1))
        layers.append(nn.Conv2d(960, 1280, 1, 1, 0, bias=False))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(1280, num_classes))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# 测试
if __name__ == "__main__":
    net = GhostNetLarge(num_classes=5)
    x = torch.randn(1, 3, 224, 224)
    y = net(x)
    print(y.shape)  # 应输出 torch.Size([1, 5])
