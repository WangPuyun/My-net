import torch
import torch.nn as nn
import torch.nn.functional as F
from attention_block import DANet_ChannelAttentionModule

class conv_block(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):

        x = self.conv(x)
        return x
    
class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class U_Net(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, in_ch=3, out_ch=1):
        super(U_Net, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

       # self.active = torch.nn.Sigmoid()

    def forward(self, x):

        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        #d1 = self.active(out)

        return out

class REBNCONV(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1):
        super(REBNCONV, self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=1 * dirate, dilation=1 * dirate)  # 空洞卷积
        self.in_s1 = nn.InstanceNorm2d(out_ch)
        self.relu_s1 = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        hx = x
        xout = self.relu_s1(self.in_s1(self.conv_s1(hx)))

        return xout

## upsample tensor 'src' to have the same spatial size with tensor 'tar'
def _upsample_like(src, tar):
    src = F.interpolate(src, size=tar.shape[2:], mode='bilinear', align_corners=True)

    return src

### RSU-7 ###
class RSU7(nn.Module):  # UNet07DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU7, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)

        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 向上取整
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv7 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv6d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)

        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)

        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)

        hx = self.pool3(hx3)
        hx4 = self.rebnconv4(hx)

        hx = self.pool4(hx4)
        hx5 = self.rebnconv5(hx)

        hx = self.pool5(hx5)
        hx6 = self.rebnconv6(hx)

        hx7 = self.rebnconv7(hx6)  # 空洞卷积可以在扩大感受野的同时不丢失分辨率

        hx6d = self.rebnconv6d(torch.cat((hx7, hx6), 1))

        hx6dup = _upsample_like(hx6d, hx5)
        hx5d = self.rebnconv5d(torch.cat((hx6dup, hx5), 1))

        hx5dup = _upsample_like(hx5d, hx4)
        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))

        hx4dup = _upsample_like(hx4d, hx3)
        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))

        hx3dup = _upsample_like(hx3d, hx2)
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))

        hx2dup = _upsample_like(hx2d, hx1)
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin

### RSU-6 ###
class RSU6(nn.Module):  # UNet06DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU6, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)

        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)

        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)

        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)

        hx = self.pool3(hx3)
        hx4 = self.rebnconv4(hx)

        hx = self.pool4(hx4)
        hx5 = self.rebnconv5(hx)

        hx6 = self.rebnconv6(hx5)

        hx5d = self.rebnconv5d(torch.cat((hx6, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin

### RSU-5 ###
class RSU5(nn.Module):  # UNet05DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU5, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)

        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)

        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)

        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)

        hx = self.pool3(hx3)
        hx4 = self.rebnconv4(hx)

        hx5 = self.rebnconv5(hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin

### RSU-4 ###
class RSU4(nn.Module):  # UNet04DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)

        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)

        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)

        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin

### RSU-4F ###
class RSU4F(nn.Module):  # UNet04FRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=4)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=8)

        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=2)

        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)

        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx2d = self.rebnconv2d(torch.cat((hx3d, hx2), 1))

        hx1d = self.rebnconv1d(torch.cat((hx2d, hx1), 1))

        return hx1d + hxin

class FinalLayer(nn.Module):
    ########  Final Layer  ########
    def __init__(self, inplanes, planes):
        super(FinalLayer, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1)
        self.In = nn.InstanceNorm2d(planes)
        self.relu = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.In(out)
        out = self.relu(out)
        out += x
        out = self.conv2(out)
        return out

##### U^2-Net ####
class AttentionU2NetOutside(nn.Module):

    def __init__(self, in_ch=8, out_ch=3):
        super(AttentionU2NetOutside, self).__init__()

        self.stage1 = RSU7(in_ch, 32, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(64, 32, 128)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(128, 64, 256)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(256, 128, 512)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(512, 256, 512)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(512, 256, 512)

        self.cam1 = DANet_ChannelAttentionModule()
        self.cam2 = DANet_ChannelAttentionModule()
        self.cam3 = DANet_ChannelAttentionModule()
        self.cam4 = DANet_ChannelAttentionModule()
        self.cam5 = DANet_ChannelAttentionModule()

        # decoder
        self.stage5d = RSU4F(1024, 256, 512)
        self.stage4d = RSU4(1024, 128, 256)
        self.stage3d = RSU5(512, 64, 128)
        self.stage2d = RSU6(256, 32, 64)
        self.stage1d = RSU7(128, 16, 64)

        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)

        self.outconv = FinalLayer(6 * out_ch, out_ch)

    def forward(self, x):
        hx = x

        # stage 1
        hx1 = self.stage1(hx)   # channel:64 256×256
        hx = self.pool12(hx1)   # channel:64 128×128

        # stage 2
        hx2 = self.stage2(hx)   # channel:128 128×128
        hx = self.pool23(hx2)   # channel:128 64×64

        # stage 3
        hx3 = self.stage3(hx)   # channel:256 64×64
        hx = self.pool34(hx3)   # channel:256 32×32

        # stage 4
        hx4 = self.stage4(hx)   # channel:512 32×32
        hx = self.pool45(hx4)   # channel:512 16×16

        # stage 5
        hx5 = self.stage5(hx)   # channel:512 16×16
        hx = self.pool56(hx5)   # channel:512 8×8

        # stage 6
        hx6 = self.stage6(hx)   # channel:512 8×8

        # -------------------- decoder --------------------
        ca5 = self.cam5(hx6)
        hx6up = _upsample_like(ca5, hx5)
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))

        ca4 = self.cam4(hx5d)
        hx5dup = _upsample_like(ca4, hx4)
        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))

        ca3 = self.cam3(hx4d)
        hx4dup = _upsample_like(ca3, hx3)
        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))

        ca2 = self.cam2(hx3d)
        hx3dup = _upsample_like(ca2, hx2)
        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))

        ca1 = self.cam1(hx2d)
        hx2dup = _upsample_like(ca1, hx1)
        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

        # side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2, d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3, d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4, d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5, d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, d1)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))

        return d0, torch.sigmoid(d1), torch.sigmoid(d2), torch.sigmoid(d3), torch.sigmoid(d4), torch.sigmoid(
            d5), torch.sigmoid(d6)
class AttentionU2Net(nn.Module):

    def __init__(self, in_ch=12, out_ch=3):
        super(AttentionU2Net, self).__init__()

        self.stage1 = RSU7(in_ch, 32, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(64, 32, 128)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(128, 64, 256)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(256, 128, 512)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(512, 256, 512)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(512, 256, 512)

        self.cam1 = DANet_ChannelAttentionModule()
        self.cam2 = DANet_ChannelAttentionModule()
        self.cam3 = DANet_ChannelAttentionModule()
        self.cam4 = DANet_ChannelAttentionModule()
        self.cam5 = DANet_ChannelAttentionModule()

        # decoder
        self.stage5d = RSU4F(1024, 256, 512)
        self.stage4d = RSU4(1024, 128, 256)
        self.stage3d = RSU5(512, 64, 128)
        self.stage2d = RSU6(256, 32, 64)
        self.stage1d = RSU7(128, 16, 64)

        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)

        self.outconv = FinalLayer(6 * out_ch, out_ch)

    def forward(self, x):
        hx = x

        # stage 1
        hx1 = self.stage1(hx)   # channel:64 256×256
        hx = self.pool12(hx1)   # channel:64 128×128

        # stage 2
        hx2 = self.stage2(hx)   # channel:128 128×128
        hx = self.pool23(hx2)   # channel:128 64×64

        # stage 3
        hx3 = self.stage3(hx)   # channel:256 64×64
        hx = self.pool34(hx3)   # channel:256 32×32

        # stage 4
        hx4 = self.stage4(hx)   # channel:512 32×32
        hx = self.pool45(hx4)   # channel:512 16×16

        # stage 5
        hx5 = self.stage5(hx)   # channel:512 16×16
        hx = self.pool56(hx5)   # channel:512 8×8

        # stage 6
        hx6 = self.stage6(hx)   # channel:512 8×8

        # -------------------- decoder --------------------
        ca5 = self.cam5(hx6)
        hx6up = _upsample_like(ca5, hx5)
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))

        ca4 = self.cam4(hx5d)
        hx5dup = _upsample_like(ca4, hx4)
        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))

        ca3 = self.cam3(hx4d)
        hx4dup = _upsample_like(ca3, hx3)
        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))

        ca2 = self.cam2(hx3d)
        hx3dup = _upsample_like(ca2, hx2)
        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))

        ca1 = self.cam1(hx2d)
        hx2dup = _upsample_like(ca1, hx1)
        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

        # side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2, d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3, d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4, d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5, d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, d1)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))

        return d0, torch.sigmoid(d1), torch.sigmoid(d2), torch.sigmoid(d3), torch.sigmoid(d4), torch.sigmoid(
            d5), torch.sigmoid(d6)

class UNet_Add_U2Net(nn.Module):

    def __init__(self, in_ch=12, out_ch=3):
        super(AttentionU2Net, self).__init__()

        self.stage1 = RSU7(in_ch, 32, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(64, 32, 128)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(128, 64, 256)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(256, 128, 512)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(512, 256, 512)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(512, 256, 512)

        self.cam1 = DANet_ChannelAttentionModule()
        self.cam2 = DANet_ChannelAttentionModule()
        self.cam3 = DANet_ChannelAttentionModule()
        self.cam4 = DANet_ChannelAttentionModule()
        self.cam5 = DANet_ChannelAttentionModule()

        # decoder
        self.stage5d = RSU4F(1024, 256, 512)
        self.stage4d = RSU4(1024, 128, 256)
        self.stage3d = RSU5(512, 64, 128)
        self.stage2d = RSU6(256, 32, 64)
        self.stage1d = RSU7(128, 16, 64)

        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)

        self.outconv = FinalLayer(6 * out_ch, out_ch)

    def forward(self, x):
        hx = x

        # stage 1
        hx1 = self.stage1(hx)   # channel:64 256×256
        hx = self.pool12(hx1)   # channel:64 128×128

        # stage 2
        hx2 = self.stage2(hx)   # channel:128 128×128
        hx = self.pool23(hx2)   # channel:128 64×64

        # stage 3
        hx3 = self.stage3(hx)   # channel:256 64×64
        hx = self.pool34(hx3)   # channel:256 32×32

        # stage 4
        hx4 = self.stage4(hx)   # channel:512 32×32
        hx = self.pool45(hx4)   # channel:512 16×16

        # stage 5
        hx5 = self.stage5(hx)   # channel:512 16×16
        hx = self.pool56(hx5)   # channel:512 8×8

        # stage 6
        hx6 = self.stage6(hx)   # channel:512 8×8

        # -------------------- decoder --------------------
        ca5 = self.cam5(hx6)
        hx6up = _upsample_like(ca5, hx5)
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))

        ca4 = self.cam4(hx5d)
        hx5dup = _upsample_like(ca4, hx4)
        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))

        ca3 = self.cam3(hx4d)
        hx4dup = _upsample_like(ca3, hx3)
        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))

        ca2 = self.cam2(hx3d)
        hx3dup = _upsample_like(ca2, hx2)
        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))

        ca1 = self.cam1(hx2d)
        hx2dup = _upsample_like(ca1, hx1)
        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

        # side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2, d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3, d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4, d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5, d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, d1)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))

        return d0, torch.sigmoid(d1), torch.sigmoid(d2), torch.sigmoid(d3), torch.sigmoid(d4), torch.sigmoid(
            d5), torch.sigmoid(d6)
def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)

net = AttentionU2NetOutside()
net.apply(initialize_weights)

if __name__ == '__main__':
    Net = AttentionU2NetOutside()
    input = torch.ones([8, 8, 256, 256])
    output, _, _, _, _, _, _, = Net(input)
    print(output.shape)
