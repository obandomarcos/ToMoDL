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
        #torch.nn.init.constant_(self.conv.weight, 0.001)
        #torch.nn.init.constant_(self.conv.bias, 0.001)
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
        :"""
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
        
        residual = torch.clone(x)    # Ojo con esto por las copias
        
        for layer in self.nw.values():

            x = layer(x)
        
        output = x + residual
        #output = residual
        
        return output


class Unet(nn.Module):
    def __init__(self, in_channel=1, out_channel=1):
        super(Unet, self).__init__()
        #Descending branch
        self.conv_encode1 = self.contract(in_channels=in_channel, out_channels=16)
        self.conv_maxpool1 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_encode2 = self.contract(16, 32)
        self.conv_maxpool2 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_encode3 = self.contract(32, 64)
        self.conv_maxpool3 = torch.nn.MaxPool2d(kernel_size=2)
        #Bottleneck
        self.bottleneck = self.bottle_neck(64)
        #Decode branch
        self.conv_decode4 = self.expans(64,64,64)
        self.conv_decode3 = self.expans(128, 64, 32)
        self.conv_decode2 = self.expans(64, 32, 16)
        self.final_layer = self.final_block(32, 16, out_channel)
        
    
    def contract(self, in_channels, out_channels, kernel_size=3, padding=1):
        block = torch.nn.Sequential(
                    torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding=padding),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm2d(out_channels),
                    torch.nn.Conv2d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels, padding=padding),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm2d(out_channels),
                    )
        return block
    
    def expans(self, in_channels, mid_channel, out_channels, kernel_size=3,padding=1):
            block = torch.nn.Sequential(
                    torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=padding),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm2d(mid_channel),
                    torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel, padding=padding),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm2d(mid_channel),
                    torch.nn.ConvTranspose2d(in_channels=mid_channel, out_channels=out_channels, kernel_size=kernel_size, stride=2,padding=padding, output_padding=1)
                    )

            return  block
    

    def concat(self, upsampled, bypass):
        out = torch.cat((upsampled,bypass),1)
        return out
    
    def bottle_neck(self,in_channels, kernel_size=3, padding=1):
        bottleneck = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=2*in_channels, padding=padding),
            torch.nn.ReLU(),
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=2*in_channels, out_channels=in_channels, padding=padding),
            torch.nn.ReLU(),
            )
        return bottleneck
    
    def final_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
            block = torch.nn.Sequential(
                    torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel,padding=1),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm2d(mid_channel),
                    torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel,padding=1),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm2d(mid_channel),
                    torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=1),
                    )
            return  block
    
    def forward(self, x ,b,c,h,w,x0, var):
        
        #Encode
        encode_block1 = self.conv_encode1(x)
        x = self.conv_maxpool1(encode_block1)
        encode_block2 = self.conv_encode2(x)
        x = self.conv_maxpool2(encode_block2)
        encode_block3 = self.conv_encode3(x)
        x = self.conv_maxpool3(encode_block3)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decode
        x = self.conv_decode4(x)
        x = self.concat(x, encode_block3)
        x = self.conv_decode3(x)
        x = self.concat(x, encode_block2)
        x = self.conv_decode2(x)
        x = self.concat(x, encode_block1)
        x = self.final_layer(x)      
        return x

class Aclass:
    """
    This class is created to do the data-consistency (DC) step as described in paper.
    """
    def __init__(self, nAngles, imageSize, mask,lam):
    
        self.mask=mask
        self.img_size = imageSize
        self.angles = np.linspace(0, 2*np.pi, nAngles,endpoint = False)
        self.det_count = int(np.sqrt(2)*self.img_size+0.5)
        self.radon = Radon(self.img_size, self.angles, clip_to_circle = False, det_count = self.det_count)
        self.lam = lam
        
    def myAtA(self,img):
        """
        Image is already in device as a Tensor
        """
        # Pending mask
        sinogram = self.radon.forward(img)
        iradon = self.radon.backprojection(self.radon.filter_sinogram(sinogram))
        del sinogram
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
         
    while((i<8) and torch.ge(rTr, 1e-6)):
            
        Ap = A.myAtA(p)
        alpha = rTr/torch.sum(p*Ap)
        x = x + alpha*p
        r = r - alpha*Ap
        rTrNew = torch.sum(r*r)
        beta = rTrNew/rTr
        p = r + beta * p
        i += 1
        rTr = rTrNew
        
    torch.cuda.empty_cache()

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
  
  def __init__(self, nLayer, K, nAngles, proj_num, imageSize, mask, lam, shared = True):
    """
    Main function that creates the model
    Params : 

        - nLayer (int) : Number of layers
        - K (int): unrolled network number of iterations
        - 

    """
    super(OPTmodl, self).__init__()
    self.out = {}
    #self.lam = torch.nn.Parameter(torch.tensor([lam], requires_grad = True, device = dev))
    self.lam = lam
    self.proj_num = proj_num
    self.epochs_save = 0

    if shared == True:
      self.dw = dw(nLayer)
    else:
      self.dw = nn.ModuleList([dw(nLayer) for _ in range(K)])
    
    if torch.cuda.is_available():
      self.dw.cuda(dev)

    self.imageSize = imageSize
    self.K = K
    self.nAngles = nAngles
    self.AtA = Aclass(nAngles, imageSize, mask, self.lam) 

  def forward(self, atb):
    """
        
    """
    # cambio linea
    self.out['dc0'] = atb
    #self.out['dw0'] = atb

    for i in range(1,self.K+1):
        # CAMBIO linea 
        j = str(i)
        self.out['dw'+j] = self.dw.forward(self.out['dc'+str(i-1)])
        rhs = atb+self.lam*self.out['dw'+j]
        self.out['dc'+j] = dc(self.AtA, rhs)
        
        # NO DC
        #self.out['dw'+str(i)] = self.dw.forward(self.out['dw'+str(i-1)])
        torch.cuda.empty_cache()
        del rhs
        # NO DW  
        #rhs = atb+self.lam*self.out['dc'+str(i-1)] 
        #self.out['dc'+str(i)] = dc(self.AtA, rhs)

    # agrego esta linea para no tener que modificar otra parte del entrenamiento
    #self.out['dc'+str(i)] = self.out['dw'+str(i)]
    self.out['dc'+j] = normalize01(self.out['dc'+j])

    return self.out

def normalize01(images):

    for img in images:
        
        img = (img - img.min())/(img.max()-img.min())

    return images


