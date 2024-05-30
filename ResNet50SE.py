import torch
import torch.nn as nn
import torch.nn.functional as F

# Squeeze-and-Excitation (SE) 层定义
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        # 适应性平均池化层用于进行通道压缩
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 两层全连接网络，第一层降维，第二层升维，用于学习通道间的非线性关系
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),  # 降维
            nn.ReLU(inplace=True),  # 非线性激活
            nn.Linear(channel // reduction, channel, bias=False),  # 升维
            nn.Sigmoid()  # 输出通道权重
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # Squeeze操作，将空间维度压缩为1x1
        y = self.fc(y).view(b, c, 1, 1)  # Excitation操作，学习每个通道的重要性
        return x * y.expand_as(x)  # 通过乘法重新调整原始特征图的通道权重

# 带有SE模块和分组卷积的Bottleneck残差块定义
class SEBottleneck(nn.Module):
    expansion = 4  # 输出通道数是输入通道数的四倍
    def __init__(self, in_planes, planes, groups, stride=1, downsample=None):
        super(SEBottleneck, self).__init__()
        # 第一个1x1卷积用于降低维度
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # 分组卷积层，减少参数数量并提高效率
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # 第三个1x1卷积用于扩展维度，恢复通道数
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes * self.expansion)  # 在最后一个卷积后添加SE模块
        self.downsample = downsample  # 如果stride不为1或者维度改变则需要调整维度的下采样层

    def forward(self, x):
        identity = x  # 保存输入，用于最后的残差连接

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.se(out)  # 应用SE模块增强特征表达

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  # 残差连接
        out = self.relu(out)
        return out

# ResNet50SE模型，使用SEBottleneck构建
class ResNet50SE(nn.Module):
    def __init__(self, block, layers, num_classes=1000, groups=32):
        super(ResNet50SE, self).__init__()
        self.in_planes = 64
        # 初始层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 构建四个残差层，不同层使用不同的配置
        self.layer1 = self._make_layer(block, 64, layers[0], groups)
        self.layer2 = self._make_layer(block, 128, layers[1], groups, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], groups, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], groups, stride=2)
        # 结尾的全局平均池化和全连接层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, groups, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, groups=groups, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.in_planes, planes, groups, stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes, groups))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# 例如，创建模型实例并指定是否预训练
model = ResNet50SE(SEBottleneck, [3, 4, 6, 3], num_classes=1000, groups=32)