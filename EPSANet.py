import torch
import torch.nn as nn
import math


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class double_conv(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(double_conv, self).__init__()
        self.layer = nn.Sequential(
            conv(in_planes, out_planes),
            nn.BatchNorm2d(out_planes),
            # 防止过拟合
            nn.Dropout(0.3),
            nn.LeakyReLU(),

            conv(out_planes, out_planes),
            nn.BatchNorm2d(out_planes),
            # 防止过拟合
            nn.Dropout(0.4),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.layer(x)


class UpSampling(nn.Module):
    def __init__(self, in_planes):
        super(UpSampling, self).__init__()
        # 特征图大小扩大2倍，通道数减半
        self.up_sampling = nn.Conv2d(in_planes, in_planes // 2, 1, 1)

    def forward(self, x1, x2):
        """
        x1: 高维准备进行上采样的张量
        x2： 低维准备进行拼接的特征
        """
        # 使用邻近插值进行下采样
        up = nn.functional.interpolate(x1, scale_factor=2, mode="nearest")
        x1 = self.up_sampling(up)
        # 拼接，当前上采样的，和之前下采样过程中的
        return torch.cat((x1, x2), dim=1)


class Up_sample(nn.Module):
    def __init__(self, in_planes):
        super(Up_sample, self).__init__()
        # 特征图大小扩大2倍，通道数减半
        self.up_sampling = nn.Conv2d(in_planes, in_planes // 2, 1, 1)

    def forward(self, x1):
        """
        x1: 高维准备进行上采样的张量
        x2： 低维准备进行拼接的特征
        """
        # 使用邻近插值进行下采样
        up = nn.functional.interpolate(x1, scale_factor=2, mode="nearest")
        x1 = self.up_sampling(up)
        # 拼接，当前上采样的，和之前下采样过程中的
        return x1


class SEWeightModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEWeightModule, self).__init__()

        self.module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # output size 1 x 1
            nn.Conv2d(channels, channels // reduction, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.module(x)


class PSAModule(nn.Module):

    def __init__(self, in_plans, out_planes, conv_kernels=[3, 5, 7, 9], stride=1, conv_groups=[1, 4, 8, 16]):
        super(PSAModule, self).__init__()
        self.conv_1 = conv(in_plans, out_planes // 4, kernel_size=conv_kernels[0], padding=conv_kernels[0] // 2,
                           stride=stride, groups=conv_groups[0])
        self.conv_2 = conv(in_plans, out_planes // 4, kernel_size=conv_kernels[1], padding=conv_kernels[1] // 2,
                           stride=stride, groups=conv_groups[1])
        self.conv_3 = conv(in_plans, out_planes // 4, kernel_size=conv_kernels[2], padding=conv_kernels[2] // 2,
                           stride=stride, groups=conv_groups[2])
        self.conv_4 = conv(in_plans, out_planes // 4, kernel_size=conv_kernels[3], padding=conv_kernels[3] // 2,
                           stride=stride, groups=conv_groups[3])
        self.se = SEWeightModule(out_planes // 4)
        self.split_channel = out_planes // 4
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # batch_size = x.shape[0]
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        x4 = self.conv_4(x)

        feats = torch.stack([x1, x2, x3, x4], dim=1)
        # feats = feats.view(batch_size, 4, self.split_channel, feats.shape[2], feats.shape[3])

        x1_se = self.se(x1)
        x2_se = self.se(x2)
        x3_se = self.se(x3)
        x4_se = self.se(x4)

        attention_vectors = torch.stack([x1_se, x2_se, x3_se, x4_se], dim=1)
        # attention_vectors = x_se.view(batch_size, 4, self.split_channel, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats_weight = feats * attention_vectors
        for i in range(4):
            x_se_weight_fp = feats_weight[:, i, :, :, :]
            if i == 0:
                out = x_se_weight_fp
            else:
                out = torch.cat((x_se_weight_fp, out), 1)

        return out


class EPSABlock(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, down_sample=None, norm_layer=None, conv_kernels=None,
                 conv_groups=None):
        super(EPSABlock, self).__init__()
        if conv_groups is None:
            conv_groups = [1, 4, 8, 16]
        if conv_kernels is None:
            conv_kernels = [3, 5, 7, 9]
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = PSAModule(planes, planes, stride=stride, conv_kernels=conv_kernels, conv_groups=conv_groups)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.down_sample = down_sample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.down_sample is not None:
            identity = self.down_sample(x)

        out += identity
        out = self.relu(out)
        return out


class EPSANet(nn.Module):
    def __init__(self, block, layers):
        super(EPSANet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layers(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layers(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layers(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layers(block, 512, layers[3], stride=2)
        # self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.up_sample_1 = UpSampling(2048)
        self.conv2 = double_conv(2048, 1024)
        self.up_sample_2 = UpSampling(1024)
        self.conv3 = double_conv(1024, 512)
        self.up_sample_3 = UpSampling(512)
        self.conv4 = double_conv(512, 256)
        self.up_sample_4 = Up_sample(256)
        self.up_sample_5 = Up_sample(128)

        self.conv5 = conv(in_planes=64, out_planes=32)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv6 = conv(in_planes=32, out_planes=32)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv7 = conv(in_planes=32, out_planes=1)   # 二分类

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. // n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layers(self, block, planes, num_blocks, stride=1):
        downsampling = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsampling = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        # (1) 构造第一个 bottleneck
        layers = [block(self.inplanes, planes, stride, downsampling)]
        self.inplanes = planes * block.expansion  # 剩余 bottleneck 的输入通道变成 planes * block.expansion

        # （2） 构造剩余的 bottleneck
        for i in range(1, num_blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # (1) Encoder
        x_0 = self.conv1(x)
        x_1 = self.bn1(x_0)
        x_2 = self.relu(x_1)
        x_3 = self.max_pool(x_2)

        x_4 = self.layer1(x_3)
        x_5 = self.layer2(x_4)
        x_6 = self.layer3(x_5)
        x_7 = self.layer4(x_6)

        # Decoder
        x_8 = self.up_sample_1(x_7, x_6)
        x_9 = self.conv2(x_8)
        x_10 = self.up_sample_2(x_9, x_5)
        x_11 = self.conv3(x_10)
        x_12 = self.up_sample_3(x_11, x_4)
        x_13 = self.conv4(x_12)
        x_14 = self.up_sample_4(x_13)
        x_15 = self.up_sample_5(x_14)   # [64, H, W]

        x_16 = self.conv5(x_15)
        x_17 = self.bn2(x_16)
        x_18 = self.conv6(x_17)
        x_19 = self.bn3(x_18)
        pred = self.conv7(x_19)

        return pred


# def epsanet_50():
#
#     return model
#
#
# def epsanet_101():
#     model = EPSANet(EPSABlock, [3, 4, 23, 3], num_classes=1000)
#     return model

# x = torch.rand([3, 3, 64, 64])
# model = EPSANet(EPSABlock, [3, 4, 6, 3])
# print(model(x).size())
