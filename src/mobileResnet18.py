import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableSEConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, reduction=16):
        super().__init__()
        # 深度卷积
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        self.se = zzbLayer(in_channels, reduction)
        # 逐点卷积
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.se(out)
        out = self.pointwise(out)
        out = self.bn(out)
        out = self.relu(out)
        return out

class zzbLayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = DepthwiseSeparableSEConv(in_planes, planes, kernel_size=3, stride=stride, padding=1)
        self.conv2 = DepthwiseSeparableSEConv(planes, planes, kernel_size=3, stride=1, padding=1)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        out = self.relu(out)
        return out

class MobileResNet(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.in_planes = 64
        self.conv1 = DepthwiseSeparableSEConv(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, planes, blocks, stride):
        downsample = None
        if stride != 1 or self.in_planes != planes:
            downsample = nn.Sequential(
                DepthwiseSeparableSEConv(self.in_planes, planes, kernel_size=1, stride=stride, padding=0),
            )
        layers = []
        layers.append(BasicBlock(self.in_planes, planes, stride, downsample))
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x