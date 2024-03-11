"""
Modified from: https://github.com/milesial/Pytorch-UNet
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def double_conv(in_channels, out_channels, mid_channels=None):
    if not mid_channels:
        mid_channels = out_channels
    return nn.Sequential(
        nn.Conv2d(in_channels, mid_channels, 3, padding=1),
        nn.BatchNorm2d(mid_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(mid_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = double_conv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        d_x = x2.size()[2] - x1.size()[2]
        d_y = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [d_y // 2, d_y - d_y // 2,
                        d_x // 2, d_x - d_x // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):

    def __init__(self, in_channels, n_classes, width):
        super().__init__()

        self.dconv_down1 = double_conv(in_channels, 64 // width)
        self.dconv_down2 = double_conv(64 // width, 128 // width)
        self.dconv_down3 = double_conv(128 // width, 256 // width)
        self.dconv_down4 = double_conv(256 // width, 512 // width)
        self.dconv_down5 = double_conv(512 // width, 1024 // width // 2)

        self.maxpool = nn.MaxPool2d(2)

        self.dconv_up5 = Up(1024 // width, 512 // width // 2)
        self.dconv_up4 = Up(512 // width, 256 // width // 2)
        self.dconv_up3 = Up(256 // width, 128 // width // 2)
        self.dconv_up2 = Up(128 // width, 64 // width)

        self.conv_last = nn.Conv2d(64 // width, n_classes, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)

        x = self.maxpool(conv1)
        conv2 = self.dconv_down2(x)

        x = self.maxpool(conv2)
        conv3 = self.dconv_down3(x)

        x = self.maxpool(conv3)
        conv4 = self.dconv_down4(x)

        x = self.maxpool(conv4)
        x = self.dconv_down5(x)

        x = self.dconv_up5(x, conv4)
        x = self.dconv_up4(x, conv3)
        x = self.dconv_up3(x, conv2)
        x = self.dconv_up2(x, conv1)
        out = self.conv_last(x)

        return out
