""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, dim=64, bilinear=True, kernel_size= 3, norm='bn'):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, dim)  # input convolution
        self.down1 = Down(dim, dim * 2, norm=norm)  # bn? out_channel=128
        self.down2 = Down(dim * 2, dim * 4, norm=norm)  # out_channel=256
        self.down3 = Down(dim * 4, dim * 8, norm=norm)  # out_channel=512
        factor = 2 if bilinear else 1
        self.down4 = Down(dim * 8, dim * 16 // factor, norm=norm)  # //向下取整
        self.up1 = Up(dim * 16, dim * 8 // factor, bilinear)  # in_channel=1024,out_channel=256 3
        self.up2 = Up(dim * 8, dim * 4 // factor, bilinear)  # in_channel=512,out_channel=128 2
        self.up3 = Up(dim * 4, dim * 2 // factor, bilinear)  # in_channel=256,out_channel=64 1
        self.up4 = Up(dim * 2, dim, bilinear)  # in_channel=128,out_channel=64 inc
        self.outc = OutConv(dim, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)  # x4>x5
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits