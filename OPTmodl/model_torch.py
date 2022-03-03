z"""
This code creates the model described in MoDL: Model-Based Deep Learning Architecture for Inverse Problems for OPT data, modifying the implementation for PyTorch in order to use Torch Radon

@author: obanmarcos
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_radon import Radon, RadonFanbeam
from torch_radon.solvers import cg
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
import torch.nn.functional as F


dev=torch.device("cuda") 

class dwLayer(nn.Module):
    """
    Creates denoiser singular layer
    """
    def __init__(self, szW, lastLayer):
        """
        Dw component initializer
        Params:
            - szW (tuple): convolutional neural network size (in_channels, out_channels, kernel_size)
            - lastLayer (bool): if True, Relu is not applied 
        """

        super().__init__()
        self.lastLayer = lastLayer
        self.conv = nn.Conv2d(*szW, padding = (int(szW[2]/2),int(szW[2]/2)))
        #torch.nn.init.constant_(self.conv.weight, 0.001)
        #torch.nn.init.constant_(self.conv.bias, 0.001)
        #self.batchNorm = nn.BatchNorm2d(szW[1])
    
    def forward(self, x):
        """
        Forward pass for block
        Params: 
            - x (torch.Tensor): Image batch to be processed
        """
        output = self.conv(x)
        #output = self.batchNorm(output)
        
        if self.lastLayer != True:
            
            output = F.relu(output)
        
        return output

class dw(nn.Module):

    def __init__(self, nLayer):
        """
        Initialises dw block
        Params:
            - nLayer (int): Number of layers 
        """
        super(dw, self).__init__()

        self.lastLayer = False
        self.nw = {}
        self.kernelSize = 3
        self.features = 64
        self.inChannels = 1
        self.outChannels = 1
        self.stride = 1

        # Intermediate layers (in_channels, out_channels, kernel_size_x, kernel_size_y)
        self.szW = {key: (self.features,self.features,self.kernelSize,self.stride) for key in range(2,nLayer)}   
        self.szW[1] = (self.inChannels, self.features, self.kernelSize, self.stride)
        self.szW[nLayer] = (self.features, self.outChannels, self.kernelSize, self.stride)

        for i in np.arange(1,nLayer+1):
            
            if i == nLayer:
                self.lastLayer = True

            self.nw['c'+str(i)] = dwLayer(self.szW[i], self.lastLayer)
            self.nw['c'+str(i)].cuda(dev)

        self.nw = nn.ModuleDict(self.nw)
          
    def forward(self, x):
        """
        Forward pass
        Params:
            - x (torch.Tensor): Image batch to be processed
        """
        residual = torch.clone(x)    
        
        for layer in self.nw.values():

            x = layer(x)
        
        output = x + residual
        
        return output

class Aclass:
    """
    This class is created to do the data-consistency (DC) step as described in paper.
    """
    def __init__(self, nAngles, imageSize, mask,lam):
    
        self.mask=mask
        self.img_size = imageSize
        self.num_angles = nAngles
        self.angles = np.linspace(0, 2*np.pi, nAngles,endpoint = False)
        self.det_count = int(np.sqrt(2)*self.img_size+0.5)
        self.radon = Radon(self.img_size, self.angles, clip_to_circle = False, det_count = self.det_count)
        self.lam = lam
        
    def myAtA(self,img):
        """
        Image is already in device as a Tensor
        """
        
        #sinogram = self.radon.forward(img)
        #iradon = self.radon.backprojection(self.radon.filter_sinogram(sinogram))

        #img = (abs(img)-abs(img).min())/(abs(img).max()-abs(img).min())         #normalize
        sinogram = self.radon.forward(img)/self.img_size    
        iradon = self.radon.backprojection(sinogram)*np.pi/self.num_angles
        #iradon = (iradon-iradon.min())/(iradon.max()-iradon.min())
        #print('img', img.max(), img.min())
        #print('iradon', iradon.max(), iradon.min())
        del sinogram
        output = iradon/self.lam+img
        
        return output
    
    def myAtA_quotient(self, img, results_folder):
       
        # Pending mask
        sinogram = self.radon.forward(img)
        iradon = self.radon.backprojection(sinogram)
        output = iradon+self.lam*img
        
        print(iradon)
        fig, ax = plt.subplots(1,1)
        ax.hist(iradon.cpu().numpy())
        ax.set_title('Iradon histogram')
        fig.savefig(results_folder+'Iradon_Hist.pdf')
        
        print(img)
        fig, ax = plt.subplots(1,1)
        ax.hist(self.lam*img.cpu().numpy())
        ax.set_title('Image histogram')
        fig.savefig(results_folder+'Image_Hist.pdf')

        fig, ax = plt.subplots(1,1)
        ax.hist(sinogram.cpu().numpy())
        ax.set_title('Sinogram histogram')
        fig.savefig(results_folder+'Sinogram_Hist.pdf')

        quotient = torch.divide(iradon, self.lam*img+1e-5)
         
        fig, ax = plt.subplots(1,1)
                                                                                             
        ax.hist(quotient.cpu().numpy())
        ax.set_title('AtA/lam*I')
        fig.savefig(results_folder+'Quotient_AtA_lamI.pdf')
         
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
         
    while((i<10) and torch.ge(rTr, 1e-6)):
        
        Ap = A.myAtA(p)
       # print('Ap', Ap.max(), Ap.min())
        alpha = rTr/torch.sum(p*Ap)
       # print('Alpha', alpha)
        x = x + alpha*p
        r = r - alpha*Ap
        rTrNew = torch.sum(r*r)
        beta = rTrNew/rTr
        p = r + beta * p
        i += 1
        rTr = rTrNew
        #print(i, rTr)

    torch.cuda.empty_cache()

    return x

def dc(Aobj, rhs, useTorchRadon = False):
    """
    Applies CG on each image on the batch
    """
    
    y = torch.zeros_like(rhs)

    for i in range(rhs.shape[0]):

        if useTorchRadon == False:
            y[i,0,:,:] = myCG(Aobj, rhs[i,0,:,:]) # This indexing may fail
        
        else:
            y[i,0,:,:] = cg(Aobj.myAtA, torch.zeros_like(rhs[i,0,:,:]), rhs[i, 0, :,:])

    return y

class OPTmodl(nn.Module):
  
  def __init__(self, nLayer, K, n_angles, proj_num, image_size, mask, lam, shared, results_folder, useUnet = False):
    """
    Main function that creates the model
    Params : 

        - nLayer (int): Number of layers
        - K (int): unrolled network number of iterations
        - n_angles (int): Number of total angles of the sinogram, fully sampled
        - proj_num (int): Number of undersampled angles of the model
        - image_size (int): Image size in pixels
        - 

    """
    super(OPTmodl, self).__init__()
    self.out = {}
    self.lam_init = lam
    self.lam = torch.nn.Parameter(torch.tensor([self.lam_init], requires_grad = True, device = dev))
    self.proj_num = proj_num
    self.epochs_save = 0
    
    if useUnet == True:
        
        self.dw = UNet(1,1)
        
    else:

        if shared == True:
            self.dw = dw(nLayer)
        else:
            self.dw = nn.ModuleList([dw(nLayer) for _ in range(K)])
    
    if torch.cuda.is_available():
      self.dw.cuda(dev)
    
    self.results_folder = results_folder
    self.print_quot = False
    self.print_epoch = 0

    self.imageSize = image_size
    self.K = K
    self.nAngles = n_angles
    self.AtA = Aclass(self.nAngles, self.imageSize, mask, self.lam) 
    
  def forward(self, atb):
    """
    Forward pass through network
    Params:
        - atb (torch.Tensor) : Backprojected sinogram, in image space    
    """
    # cambio linea
    self.out['dc0'] = atb
    #self.out['dw0'] = atb

    for i in range(1,self.K+1):
    
        j = str(i)
        self.out['dw'+j] = self.dw.forward(self.out['dc'+str(i-1)])
        rhs = atb/self.lam+self.out['dw'+j]

        self.out['dc'+j] = dc(self.AtA, rhs)
        
        torch.cuda.empty_cache()
        del rhs

    self.out['dc'+j] = normalize01(self.out['dc'+j])    
    #print('output network max', self.out['dc'+j].max(), 'min', self.out['dc'+j].min()) 
    return self.out
  
  def plot_histogram(self,x):

    fig, ax = plt.subplots(1,1)
    
    title = 'Atb/lam*Zn'
    ax.hist(x)
    ax.set_title(title)
    fig.savefig(self.results_folder+'Quotient_Atb_lamZn_{}.pdf'.format(self.print_epoch))

def normalize01(images):
    
    image_norm = torch.zeros_like(images)

    for i, img in enumerate(images):
         
        image_norm[i,...] = (img - img.min())/(img.max()-img.min())

    return image_norm

# U-Net

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
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

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            double_conv(in_ch, out_ch)
        )
    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

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
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
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
        return x#F.sigmoid(x)




























































































