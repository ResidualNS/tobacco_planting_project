""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnReLU(nn.Sequential):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels,kernel_size=3,stride=1,padding=1,dilation=1,relu=True):
        super().__init__()
        self.add_module("conv",nn.Conv2d(in_channels, out_channels, kernel_size=3,stride=1,padding=1,dilation=1,bias=False),)
        self.add_module("bn",nn.BatchNorm2d(out_channels,eps=1e-5,momentum=0.999))
        if relu:
            self.add_module("relu",nn.ELU())

#bottlenneck
class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels,strides=1,dilation=1,downsample=False):
        super().__init__()
        mid_channels=out_channels//4
        self.reduce=ConvBnReLU(in_channels,mid_channels,kernel_size=1,stride=1,padding=0,relu=True)
        self.conv33=ConvBnReLU(mid_channels,mid_channels,kernel_size=3,stride=1,padding=1,relu=True)
        self.increase=ConvBnReLU(mid_channels,out_channels,kernel_size=1,stride=1,padding=0,relu=False)
        self.shortcut=(ConvBnReLU(in_channels,out_channels,kernel_size=1,stride=1,padding=0,relu=False) if downsample else lambda x:x)

    def forward(self, x):
        x=self.reduce(x)
        x=self.conv33(x)
        x=self.increase(x)
        x=self.shortcut(x)
        return F.leaky_relu(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            Bottleneck(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = Bottleneck(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = Bottleneck(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
