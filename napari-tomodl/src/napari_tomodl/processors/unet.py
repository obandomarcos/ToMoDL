'''
This code creates the model described in 'Convolutional neural networks for reconstruction of undersampled optical projection tomography data applied to in vivo imaging of zebrafish' and derived from https://github.com/imperial-photonics/CNOPT

author: obanmarcos 
'''

import torch
import torch.nn as nn
import numpy as np
from torch_radon import Radon, RadonFanbeam
from torch_radon.solvers import cg
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
import torch.nn.functional as F


# Modify for multi-gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# U-Net

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
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
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1, padding = (2,2))

    def forward(self, x):
        x = self.conv(x)
        return x

class UNet(nn.Module):
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
        x = self.outc(x)
        if self.residual is True:        
            x = x+self.lam*x0

        return x#F.sigmoid(x)