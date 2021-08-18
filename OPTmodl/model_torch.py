"""
This code creates the model described in MoDL: Model-Based Deep Learning Architecture for Inverse Problems for OPT data, modifying the implementation for PyTorch in order to use Torch Radon

@author: obanmarcos
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class dwLayer(nn.Module):
    
    def __init__(self, szW, lastLayer):
        """
        Dw component
        """
        self.lastLayer = lastLayer
        self.conv = nn.Conv2d(szW)
        self.batch = nn.BatchNorm2d(szW[1])
    
    def forward(self, x):
        
        self.
        


class dw(nn.Module):

    def __init__(self, nLayer):
        """
        Initialises dw block
        """
        super(dw, self).__init__()
        self.lastLayer = False
        self.nw = {}
        self.kernelSize = 3
        self.features = 64
        self.inChannels = 1
        self.outChannels = 1
        self.szW = {key: (self.features,self.features,self.kernelSize,self.kernelSize) for key in range(2,nLayer)}   # Intermediate layers (in_channels, out_channels, kernel_size_x, kernel_size_y)
        self.szW[1] = (self.inChannels, self.features, self.kernelSize, self.kernelSize)
        self.szW[nLayer] = (self.features, self.outChannels, self.kernelSize, self.kernelSize)

        for i in np.arange(1,nLayer+1):
            
            self.nw['c'+str(i)] = {'conv':nn.Conv2d(*self.szW[i]),
                                    'batch_norm': nn.BatchNorm2d(self.szW[i][1])}
            if i != nLayer:
                self.nw['c'+str(i)]['relu'] = 
            

