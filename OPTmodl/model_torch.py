"""
This code creates the model described in MoDL: Model-Based Deep Learning Architecture for Inverse Problems for OPT data, modifying the implementation for PyTorch in order to use Torch Radon

@author: obanmarcos
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_radon import Radon, RadonFanbeam

dev=torch.device("cuda") 

class dwLayer(nn.Module):
    
    def __init__(self, szW, lastLayer):
        """
        Dw component
        """
        super().__init__()
        self.lastLayer = lastLayer
        self.conv = nn.Conv2d(*szW, padding = (int(szW[2]/2),int(szW[2]/2)))
        self.batchNorm = nn.BatchNorm2d(szW[1])
    
    def forward(self, x):
        """
        Forward pass for block
        """
        x = self.conv(x)
        output = self.batchNorm(x)
        
        if self.lastLayer != True:
            
            output = F.relu(output)
        
        return output

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
        self.stride = 1
        self.szW = {key: (self.features,self.features,self.kernelSize,self.stride) for key in range(2,nLayer)}   # Intermediate layers (in_channels, out_channels, kernel_size_x, kernel_size_y)
        self.szW[1] = (self.inChannels, self.features, self.kernelSize, self.stride)
        self.szW[nLayer] = (self.features, self.outChannels, self.kernelSize, self.stride)

        for i in np.arange(1,nLayer+1):
            
            if i == nLayer:
                self.lastLayer = True

            self.nw['c'+str(i)] = dwLayer(self.szW[i], self.lastLayer)
            self.nw['c'+str(i)].cuda(dev)

        self.nw = nn.ModuleDict(self.nw)
          
    def forward(self, x):
        
        residual = x    # Ojo con esto por las copias
        
        for layer in self.nw.values():

            x = layer(x)
        
        output = x + residual

        return output
 
class Aclass:
    """
    This class is created to do the data-consistency (DC) step as described in paper.
    """
    def __init__(self, maxAngle, nAngles, imageSize, mask,lam):
    
        self.mask=mask
        self.imageSize = imageSize
        self.angles = np.linspace(0,maxAngle,nAngles,endpoint = False)
        self.det_count = int(np.sqrt(2)*self.imageSize+0.5)
        self.radon = Radon(self.imageSize, self.angles, clip_to_circle = False, det_count = self.det_count)
        self.lam = lam
        
    def myAtA(self,img):
        """
        Image is already in device as a Tensor
        """
        # Pending mask
        sinogram = self.radon.forward(img)
        iradon = self.radon.backward(self.radon.filter_sinogram(sinogram))
        output = iradon+self.lam*img

        return output

def myCG(A,rhs):
    """
    My implementation of conjugate gradients in PyTorch
    """
    
    i = 0
    x = torch.zeros_like(rhs)
    r = rhs 
    p = rhs 
    rTr = torch.sum(r*r)

    while((i<10) or (rTr<1e-10)):

        Ap = A.myAtA(p)
        alpha = rTr/torch.sum(p*Ap)
        x = x + alpha*p
        r = r - alpha*Ap
        rTrNew = torch.sum(r*r)
        beta = rTrNew/rTr
        p = r + beta * p
        i += 1
        rTr = rTrNew
        
    return x

def dc(Aobj, rhs):
    """
    Applies CG on each image on the batch
    """
    
    y = torch.zeros_like(rhs)

    for i in range(rhs.shape[0]):

        y[i,0,:,:] = myCG(Aobj, rhs[i,0,:,:]) # This indexing may fail

    return y

class OPTmodl(nn.Module):
  
  def __init__(self, nLayer, K, maxAngle, nAngles, imageSize, mask, lam, shared = True):
    """
    Main function that creates the model
    """
    super(OPTmodl, self).__init__()
    self.out = {}
    self.lam = lam

    if shared == True:
      self.dw = dw(nLayer)
    else:
      self.dw = nn.ModuleList([dw(nLayer) for _ in range(K)])
    
    if torch.cuda.is_available():
      self.dw.cuda(dev)

    self.imageSize = imageSize
    self.K = K
    self.AtA = Aclass(maxAngle, nAngles, imageSize, mask, self.lam) 

  def forward(self, atb):
    """
    - Reusing weights for pytorch seems o be different than TF
    """
    self.out['dc0'] = atb

    for i in range(1,self.K+1):
        
        j = str(i)
        self.out['dw'+j] = self.dw.forward(self.out['dc'+str(i-1)])
        rhs = atb+self.lam*self.out['dw'+j]
        self.out['dc'+j] = dc(self.AtA, rhs)
        # self.out['dc'+j] = rhs
        
    return self.out
