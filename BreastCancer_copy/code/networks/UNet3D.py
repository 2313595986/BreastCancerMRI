import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.doule_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.doule_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(kernel_size=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = DoubleConv(int(in_channels/2 + in_channels), out_channels, in_channels//2)

    def forward(self, x1, x2):
        # x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet3D(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet3D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 32 * 2)
        self.down2 = Down(32 * 2, 32 * 4)
        self.down3 = Down(32 * 4, 32 * 8)
        self.down4 = Down(32 * 8, 320)
        self.up1 = Up(320, 8 * 32)
        self.up2 = Up(8 * 32, 4 * 32)
        self.up3 = Up(4 * 32, 2 * 32)
        self.up3 = Up(2 * 32, 32)
        # self.outc = OutConv(8, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        y1 = self.up1(x4, x3)
        y2 = self.up2(y1, x2)
        y3 = self.up3(y2, x1)
        logits = self.outc(y3)

        return [logits]


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(2, 2, 2), stride=(2, 2, 2)):
        super(Upsample, self).__init__()
        self.ConvTrans = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, bias=False)
        self.norm = nn.InstanceNorm3d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)

    def forward(self, x):
        return self.norm(self.ConvTrans(x))

class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, input):
        return input

class ConvDropoutNormReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), is_dropout=False):
        super(ConvDropoutNormReLU, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding=[(i - 1) // 2 for i in kernel_size])
        self.norm = nn.InstanceNorm3d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        self.nonlin = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        if is_dropout:
            self.dropout = nn.Dropout3d(p=0.2, inplace=True)
        else:
            self.dropout = Identity()

        self.all = nn.Sequential(self.conv, self.dropout, self.norm, self.nonlin)

    def forward(self, x):
        return self.all(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3),
                 conv_stride=(1, 1, 1), is_dropout=False):
        super(DecoderBlock, self).__init__()
        self.conv = ConvDropoutNormReLU(in_channels, out_channels, kernel_size, conv_stride, is_dropout)
    def forward(self, x):
        return self.conv(x)


class UNet3D_DS(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet3D_DS, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 32 * 2)
        self.down2 = Down(32 * 2, 32 * 4)
        self.down3 = Down(32 * 4, 32 * 8)
        self.down4 = Down(32 * 8, 320)
        self.up1 = Upsample(320, 8 * 32, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.up2 = Upsample(8 * 32, 4 * 32, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.up3 = Upsample(4 * 32, 2 * 32, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.up4 = Upsample(2 * 32, 1 * 32, kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.decoder1 = DecoderBlock(512, 256)
        self.decoder2 = DecoderBlock(256, 128)
        self.decoder3 = DecoderBlock(128, 64)
        self.decoder4 = DecoderBlock(64, 32)

        self.out1 = nn.Conv3d(256, 2, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.out2 = nn.Conv3d(128, 2, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.out3 = nn.Conv3d(64, 2, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.out4 = nn.Conv3d(32, 2, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x5_up = self.up1(x5)
        x4_x5 = torch.cat((x4, x5_up), dim=1)
        y1 = self.decoder1(x4_x5)

        y1_up = self.up2(y1)
        x3_y1 = torch.cat((x3, y1_up), dim=1)
        y2 = self.decoder2(x3_y1)

        y2_up = self.up3(y2)
        x2_y2 = torch.cat((x2, y2_up), dim=1)
        y3 = self.decoder3(x2_y2)

        y3_up = self.up4(y3)
        x1_y3 = torch.cat((x1, y3_up), dim=1)
        y4 = self.decoder4(x1_y3)

        out1 = self.out1(y1)
        out2 = self.out2(y2)
        out3 = self.out3(y3)
        out4 = self.out4(y4)

        out_list = [out4, out3, out2, out1]
        return out_list


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '5'
    model = UNet3D(1, 2).cuda()
    input = torch.randn((1, 1, 192, 96, 64)).cuda()
    out = model(input)
