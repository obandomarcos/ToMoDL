"""
U-Net architecture and helper layers for tomographic image reconstruction.

This module implements:
    • A configurable U-Net for denoising or learned reconstruction
    • Optional batch normalization, dropout, bilinear upsampling, and residual output
    • Supporting building blocks: double convolution, downsampling, upsampling, and output layer

The implementation is compatible with GPU inference and can be integrated into
model-based reconstruction frameworks such as MoDL or ToMoDL.

References:
    - Ronneberger et al., “U-Net: Convolutional Networks for Biomedical Image Segmentation”, MICCAI 2015.
    - Adapted for use in OPT / CT tomographic deep reconstruction pipelines.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


try:
    # Modify for multi-gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

except:
    print('Torch not available!')
    
# U-Net

class double_conv(nn.Module):
    """Two consecutive convolution layers with optional batch normalization.

    Each block performs:
        Conv → (BatchNorm) → ReLU → Dropout →
        Conv → (BatchNorm) → ReLU → Dropout

    Args:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
        batch_norm (bool): Whether to include BatchNorm2d layers.

    Forward Args:
        x (Tensor): Input tensor of shape (B, C, H, W).

    Returns:
        Tensor: Output feature map after double convolution.
    """
    def __init__(self, in_ch, out_ch, batch_norm = False):
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
                nn.Dropout2d()
            )
        else:

            self.conv = nn.Sequential(
              nn.Conv2d(in_ch, out_ch, 3, padding=1),
              nn.ReLU(inplace=True),
              nn.Dropout2d(),
              nn.Conv2d(out_ch, out_ch, 3, padding=1),
              nn.ReLU(inplace=True),
              nn.Dropout2d()
            )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    """Initial convolution block of the U-Net encoder.

    Wraps `double_conv` with optional batch normalization on the input.

    Args:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
        batch_norm (bool): Apply BatchNorm before convolutions.

    Forward Args:
        x (Tensor): Input batch.

    Returns:
        Tensor: First feature map of the U-Net encoder.
    """
    def __init__(self, in_ch, out_ch, batch_norm):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch, batch_norm = batch_norm)
        
        if batch_norm == True:

            self.conv = nn.Sequential(
                nn.BatchNorm2d(in_ch),
                double_conv(in_ch, out_ch)
            )
#        else :
#            self.conv = nn.Sequential(double_conv(in_ch, out_ch))

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    """Downsampling block: MaxPool2d followed by double_conv.

    Args:
        in_ch (int): Input channels.
        out_ch (int): Output channels.
        batch_norm (bool): Enable batch normalization in sub-convolutions.

    Forward Args:
        x (Tensor): Input tensor.

    Returns:
        Tensor: Downsampled feature map.
    """
    def __init__(self, in_ch, out_ch, batch_norm):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch, batch_norm)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    """Upsampling block using either bilinear upsampling or transposed convolution.

    Upsamples x1, aligns it with skip-connection x2, concatenates along channel
    dimension, and applies a double_conv block.

    Args:
        in_ch (int): Number of channels in concatenated input.
        out_ch (int): Number of output channels after convolution.
        bilinear (bool): If True, use bilinear Upsample; otherwise ConvTranspose2d.
        batch_norm (bool): Apply batch normalization in convolution layers.

    Forward Args:
        x1 (Tensor): Decoder feature map (to be upsampled).
        x2 (Tensor): Skip-connection feature map from encoder.

    Returns:
        Tensor: Fused decoder feature map.
    """
    def __init__(self, in_ch, out_ch, bilinear=True, batch_norm = False):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch, batch_norm)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    """Final 1×1 convolution layer producing network output.

    Pads input features to match the spatial size of the skip-origin (x0).

    Args:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output classes/channels.

    Forward Args:
        x0 (Tensor): Original input (for spatial alignment).
        x (Tensor): Final decoder output.

    Returns:
        Tensor: Output image/map of shape (B, out_ch, H, W).
    """
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1, padding = "same")

    def forward(self, x0, x):
        # padding to x.shape == x0.shape    
        diffX = x0.size()[2] - x.size()[2]
        diffY = x0.size()[3] - x.size()[3]
        x = F.pad(x, (diffX // 2, int(diffX / 2), diffY // 2, int(diffY / 2)))
        x = self.conv(x)
        return x

class UNet(nn.Module):
    """Standard U-Net with optional residual output connection.

    Architecture:
        Encoder:
            inconv → down1 → down2 → down3 → down4
        Decoder:
            up1 → up2 → up3 → up4 → outconv

    Features:
        • Optional bilinear upsampling
        • Optional batch normalization
        • Optional residual output:  output = U(x) + λx
        • Suitable for denoising, deblurring, and tomographic reconstruction

    Args:
        n_channels (int): Number of input channels (e.g., 1 for grayscale).
        n_classes (int): Number of output channels.
        up_conv (bool): Use bilinear upsampling instead of ConvTranspose2d.
        residual (bool): Enable residual output (U(x) + λx).
        batch_norm (bool): Apply batch normalization in encoder/decoder.
        batch_norm_inconv (bool): Apply batch normalization in first layer.

    Forward Args:
        x0 (Tensor): Input image batch of shape (B, C, H, W).

    Returns:
        Tensor: Reconstructed/denoised image(s).
    """
    def __init__(self, n_channels, n_classes, up_conv = False, residual = False, batch_norm = False, batch_norm_inconv = False):
        
        super(UNet, self).__init__()
        
        self.residual = residual
        if self.residual is True:
            self.lam = torch.nn.Parameter(torch.tensor([0.1], requires_grad = True, device = device))
        self.inc = inconv(n_channels, 64, batch_norm = batch_norm_inconv)
        self.down1 = down(64, 128, batch_norm = batch_norm)
        self.down2 = down(128, 256, batch_norm = batch_norm)
        self.down3 = down(256, 512,  batch_norm = batch_norm)
        self.down4 = down(512, 512, batch_norm = batch_norm)
        self.up1 = up(1024, 256, bilinear = up_conv, batch_norm = batch_norm)
        self.up2 = up(512, 128, bilinear = up_conv, batch_norm = batch_norm)
        self.up3 = up(256, 64, bilinear = up_conv, batch_norm = batch_norm)
        self.up4 = up(128, 64 , bilinear = up_conv, batch_norm = batch_norm)
        self.outc = outconv(64, n_classes)
    
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
        x = self.outc(x0, x)

        if self.residual is True:        
            x = x+self.lam*x0

        return x#F.sigmoid(x)