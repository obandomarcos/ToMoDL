"""
This code creates the model described in 'Convolutional neural networks for reconstruction of undersampled optical projection tomography data applied to in vivo imaging of zebrafish' and derived from https://github.com/imperial-photonics/CNOPT

author: obanmarcos
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

# Modify for multi-gpu
# U-Net


class double_conv(nn.Module):
    """(conv => BN => ReLU) * 2"""

    def __init__(self, in_ch, out_ch, batch_norm=False):
        super(double_conv, self).__init__()

        if batch_norm == True:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Dropout2d(),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Dropout2d(),
            )
        else:

            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Dropout2d(),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Dropout2d(),
            )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, batch_norm):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch, batch_norm=batch_norm)

        if batch_norm == True:

            self.conv = nn.Sequential(nn.BatchNorm2d(in_ch), double_conv(in_ch, out_ch))

    #        else :
    #            self.conv = nn.Sequential(double_conv(in_ch, out_ch))

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch, batch_norm):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(nn.MaxPool2d(2), double_conv(in_ch, out_ch, batch_norm))

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True, batch_norm=False):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch, batch_norm)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2), diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1, padding=(2, 2))

    def forward(self, x):
        x = self.conv(x)
        return x


class unet(nn.Module):
    def __init__(self, kw_dict):

        super(unet, self).__init__()

        self.process_kwdictionary(kw_dict)

        if self.residual is True:
            self.lam = torch.nn.Parameter(torch.tensor([0.1], requires_grad=True, device=device))

        self.inc = inconv(self.n_channels, 64, batch_norm=self.batch_norm_inconv)
        self.down1 = down(64, 128, batch_norm=self.batch_norm)
        self.down2 = down(128, 256, batch_norm=self.batch_norm)
        self.down3 = down(256, 512, batch_norm=self.batch_norm)
        self.down4 = down(512, 512, batch_norm=self.batch_norm)
        self.up1 = up(1024, 256, bilinear=self.up_conv, batch_norm=self.batch_norm)
        self.up2 = up(512, 128, bilinear=self.up_conv, batch_norm=self.batch_norm)
        self.up3 = up(256, 64, bilinear=self.up_conv, batch_norm=self.batch_norm)
        self.up4 = up(128, 64, bilinear=self.up_conv, batch_norm=self.batch_norm)
        self.outc = outconv(64, self.n_classes)

    def forward(self, x0):

        x1 = self.inc(x0)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        if self.residual is True:
            x = x + self.lam * x0

        return x  # F.sigmoid(x)

    def process_kwdictionary(self, kw_unet_dict):
        """
        Process KW dictionary
        """

        self.n_channels = kw_unet_dict["n_channels"]
        self.n_classes = kw_unet_dict["n_classes"]
        self.bilinear = kw_unet_dict["bilinear"]
        self.batch_norm = kw_unet_dict["batch_norm"]
        self.batch_norm_inconv = kw_unet_dict["batch_norm_inconv"]
        self.residual = kw_unet_dict["residual"]
        self.up_conv = kw_unet_dict["up_conv"]


# https://github.com/miguelvr/dropblock/blob/master/dropblock/dropblock.py
class DropBlock2D(nn.Module):
    def __init__(self, drop_prob, block_size):
        super(DropBlock2D, self).__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        # shape: (bsize, channels, height, width)
        assert x.dim() == 4, "Expected input with 4 dimensions (bsize, channels, height, width)"
        if not self.training or self.drop_prob == 0.0:
            return x
        else:
            # get gamma value
            gamma = self.drop_prob / (self.block_size**2)
            # sample mask
            mask = (torch.rand(x.shape[0], *x.shape[2:], device=x.device) < gamma).float()
            # compute block mask
            block_mask = self._compute_block_mask(mask)
            # apply block mask
            out = x * block_mask[:, None, :, :]
            # scale output
            out = out * block_mask.numel() / block_mask.sum()
            return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool2d(
            input=mask[:, None, :, :],
            kernel_size=(self.block_size, self.block_size),
            stride=(1, 1),
            padding=self.block_size // 2,
        )

        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]
        block_mask = 1 - block_mask.squeeze(1)
        return block_mask


def match_size(x, ref):
    if x.shape[-2:] != ref.shape[-2:]:
        x = F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
    return x


class conv_block(nn.Sequential):
    def __init__(self, ch_in, ch_out, kernel_size=3, padding=1, drop_block=False, block_size=1, drop_prob=0):
        super().__init__()
        self.add_module("conv1", nn.Conv2d(ch_in, ch_out, kernel_size, padding=padding, bias=False))
        self.add_module("bn1", nn.GroupNorm(1, ch_out))
        self.add_module("relu1", nn.ReLU(inplace=True))
        self.add_module("conv2", nn.Conv2d(ch_out, ch_out, kernel_size, padding=padding, bias=False))
        if drop_block:
            self.add_module("drop_block", DropBlock2D(block_size=block_size, drop_prob=drop_prob))
        self.add_module("bn2", nn.GroupNorm(1, ch_out))
        self.add_module("relu2", nn.ReLU(inplace=True))


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(
                ch_in,
                ch_out,
                kernel_size=2,
                stride=1,
                padding="same",
                bias=False,
            ),
            nn.GroupNorm(1, ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True), nn.GroupNorm(1, F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True), nn.GroupNorm(1, F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True), nn.GroupNorm(1, 1), nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class AttU_Net(nn.Module):
    def __init__(self, img_ch=1, output_ch=1, drop_prob=0.1):
        super(AttU_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        self.Conv1 = conv_block(ch_in=img_ch, ch_out=16)
        self.Conv2 = conv_block(ch_in=16, ch_out=32)
        self.Conv3 = conv_block(ch_in=32, ch_out=64)
        self.Conv4 = conv_block(
            ch_in=64,
            ch_out=128,
            drop_block=True,
            block_size=5,
            drop_prob=drop_prob,
        )
        self.Conv5 = conv_block(
            ch_in=128,
            ch_out=256,
            drop_block=True,
            block_size=5,
            drop_prob=drop_prob,
        )

        # Decoder
        self.Up5 = up_conv(ch_in=256, ch_out=128)
        self.Att5 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_conv5 = conv_block(ch_in=256, ch_out=128)

        self.Up4 = up_conv(ch_in=128, ch_out=64)
        self.Att4 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_conv4 = conv_block(ch_in=128, ch_out=64)

        self.Up3 = up_conv(ch_in=64, ch_out=32)
        self.Att3 = Attention_block(F_g=32, F_l=32, F_int=16)
        self.Up_conv3 = conv_block(ch_in=64, ch_out=32)

        self.Up2 = up_conv(ch_in=32, ch_out=16)
        self.Att2 = Attention_block(F_g=16, F_l=16, F_int=8)
        self.Up_conv2 = conv_block(ch_in=32, ch_out=16)

        self.Conv_1x1 = nn.Conv2d(
            in_channels=16,
            out_channels=output_ch,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x):
        # Encoder
        x1 = self.Conv1(x)  # [B, 16, H, W]

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)  # [B, 32, H/2, W/2]

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)  # [B, 64, H/4, W/4]

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)  # [B, 128, H/8, W/8]

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)  # [B, 256, H/16, W/16]

        # Decoder
        d5 = self.Up5(x5)  # [B, 128, H/8, W/8]
        d5 = match_size(d5, x4)  # Ensure d5 matches x4's size
        x4_att = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4_att, d5), dim=1)  # [B, 256, H/8, W/8]
        d5 = self.Up_conv5(d5)  # [B, 128, H/8, W/8]

        d4 = self.Up4(d5)  # [B, 64, H/4, W/4]
        d4 = match_size(d4, x3)  # Ensure d4 matches x3's size
        x3_att = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3_att, d4), dim=1)  # [B, 128, H/4, W/4]
        d4 = self.Up_conv4(d4)  # [B, 64, H/4, W/4]

        d3 = self.Up3(d4)  # [B, 32, H/2, W/2]
        d3 = match_size(d3, x2)  # Ensure d3 matches x2's size
        x2_att = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2_att, d3), dim=1)  # [B, 64, H/2, W/2]
        d3 = self.Up_conv3(d3)  # [B, 32, H/2, W/2]

        d2 = self.Up2(d3)  # [B, 16, H, W]
        d2 = match_size(d2, x1)  # Ensure d2 matches x1's size
        x1_att = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1_att, d2), dim=1)  # [B, 32, H, W]
        d2 = self.Up_conv2(d2)  # [B, 16, H, W]

        out = self.Conv_1x1(d2)  # [B, output_ch, H, W]

        return out
