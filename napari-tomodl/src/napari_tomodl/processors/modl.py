"""
This code creates the model described in MoDL: Model-Based Deep Learning Architecture for Inverse Problems for OPT data, modifying the implementation for PyTorch in order to use Torch Radon

@author: obanmarcos
"""
import torch
import torch.nn as nn
import numpy as np
try:
    from torch_radon import Radon as thrad
except ImportError:
    pass

from skimage.transform import radon, iradon

from torch_radon.solvers import cg
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import unet

# Modify for multi-gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class dwLayer(nn.Module):
    """
    Creates denoiser singular layer
    """
    def __init__(self, kw_dictionary):
        """
        Dw component initializer
        Params:
            - weights_size (tuple): convolutional neural network size (in_channels, out_channels, kernel_size)
            - is_last_layer (bool): if True, Relu is not applied 
            - init_method (string): Initialization method, defaults to Xavier init
        """

        super().__init__()

        self.process_kwdictionary(kw_dictionary)
        self.conv = nn.Conv2d(*self.weights_size, padding = (int(self.weights_size[2]/2),int(self.weights_size[2]/2)))

        self.initialize_layer(method = self.init_method)            
        
        if self.use_batch_norm == True:
            self.batch_norm = nn.BatchNorm2d(self.weights_size[1])
    
    def forward(self, x):
        """
        Forward pass for block
        Params: 
            - x (torch.Tensor): Image batch to be processed
        """
        output = self.conv(x)

        if self.use_batch_norm:

            output = self.batch_norm(output)
        
        if self.is_last_layer != True:
            
            output = F.relu(output)
        
        return output
    
    def process_kwdictionary(self, kw_dictionary):
        '''
        Process keyword dictionary.
        Params: 
            - kw_dictionary (dict): Dictionary with keywords
        '''

        self.weights_size = kw_dictionary['weights_size']
        self.is_last_layer = kw_dictionary['is_last_layer']
        self.init_method = kw_dictionary['init_method']
        self.use_batch_norm = kw_dictionary['use_batch_norm']

    def initialize_layer(self, method):
        '''
        Initializes convolutional weights according to method
        Params:
         - method (string): Method of initialization, please refer to https://pytorch.org/docs/stable/nn.init.html
        '''
        if method == 'xavier':
            return
        elif method == 'constant':
            torch.nn.init.constant_(self.conv.weight, 0.001)
            torch.nn.init.constant_(self.conv.bias, 0.001)

class dw(nn.Module):

    def __init__(self, kw_dictionary):
        """
        Initialises dw block
        Params:
            - kw_dictionary (dict): Parameters dictionary
        """
        super(dw, self).__init__()

        self.process_kwdictionary(kw_dictionary=kw_dictionary)
        
        for i in np.arange(1, self.number_layers+1):
            
            self.dw_layer_dict['weights_size'] = self.weights_size[i]

            if i == self.number_layers-1:
                self.dw_layer_dict['is_last_layer']= True

            self.nw['c'+str(i)] = dwLayer(self.dw_layer_dict)
            self.nw['c'+str(i)].cuda(device)

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

    def process_kwdictionary(self, kw_dictionary):
        '''
        Process keyword dictionary.
        Params: 
            - kw_dictionary (dict): Dictionary with keywords
        '''
        
        self.number_layers = kw_dictionary['number_layers']
        self.nw = {}
        self.kernel_size = kw_dictionary['kernel_size']
        self.features = kw_dictionary['features']
        self.in_channels = kw_dictionary['in_channels']
        self.out_channels= kw_dictionary['out_channels']
        self.stride = kw_dictionary['stride']
        self.use_batch_norm = kw_dictionary['use_batch_norm']
        self.init_method = kw_dictionary['init_method']

        # Intermediate layers (in_channels, out_channels, kernel_size_x, kernel_size_y)
        self.weights_size = {key: (self.features,self.features,self.kernel_size,self.stride) for key in range(2, self.number_layers)}   
        self.weights_size[1] = (self.in_channels, self.features, self.kernel_size, self.stride)
        self.weights_size[self.number_layers] = (self.features, self.out_channels, self.kernel_size, self.stride)
        
        self.dw_layer_dict = {'use_batch_norm':self.use_batch_norm,
        'is_last_layer': False,
        'init_method':self.init_method}

class ToMoDL(nn.Module):
  
  def __init__(self, kw_dictionary):
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
    super(ToMoDL, self).__init__()

    self.process_kwdictionary(kw_dictionary)
    self.define_denoiser()
    
  def forward(self, x):
    """
    Forward pass through network
    Params:
        - x (torch.Tensor) : Backprojected sinogram, in image space    
    """
    
    self.out['dc0'] = x

    for i in range(1,self.K+1):
    
        j = str(i)
        
        self.out['dw'+j] = normalize_images(self.dw.forward(self.out['dc'+str(i-1)]))
        rhs = x/self.lam+self.out['dw'+j]

        self.out['dc'+j] = normalize_images(self.AtA.inverse(rhs))
        
        del rhs

        torch.cuda.empty_cache()

    return self.out
  
  def process_kwdictionary(self, kw_dictionary):
    '''
    Process keyword dictionary.
    Params: 
        - kw_dictionary (dict): Dictionary with keywords
    '''

    self.out = {}
    self.use_torch_radon = kw_dictionary['use_torch_radon']
    self.K = kw_dictionary['K_iterations']
    self.number_projections_total = kw_dictionary['number_projections_total']
    self.acceleration_factor = kw_dictionary['acceleration_factor']
    self.number_projections_undersampled = self.number_projections_total//self.acceleration_factor
    self.image_size = kw_dictionary['image_size'] 
    
    self.lam = kw_dictionary['lambda']
    self.lam = torch.nn.Parameter(torch.tensor([self.lam], requires_grad = True, device = device))
    
    self.use_shared_weights = kw_dictionary['use_shared_weights']
    self.denoiser_method = kw_dictionary['denoiser_method']
    
    self.in_channels = kw_dictionary['in_channels']
    self.out_channels = kw_dictionary['out_channels']

    if self.denoiser_method == 'U-Net':
        self.unet_options = kw_dictionary['unet_options']
    elif self.denoiser_method == 'resnet':
        self.resnet_options = kw_dictionary['resnet_options']
    
    self.AtA_dictionary = {'image_size': self.image_size, 'number_projections': self.number_projections_total, 'lambda':self.lam, 'use_torch_radon': self.use_torch_radon}

    self.AtA = Aclass(self.AtA_dictionary)

  def define_denoiser(self):
    '''
    Defines denoiser used in MoDL. Options include Resnet and U-Net

    References:
        - Aggarwal, H. K., Mani, M. P., & Jacob, M. (2018). MoDL: Model-based deep learning architecture for inverse problems. IEEE transactions on medical imaging, 38(2), 394-405.
        - Davis, S. P., Kumar, S., Alexandrov, Y., Bhargava, A., da Silva Xavier, G., Rutter, G. A., ... & McGinty, J. (2019). Convolutional neural networks for reconstruction of undersampled optical projection tomography data applied to in vivo imaging of zebrafish. Journal of biophotonics, 12(12), e201900128.
    '''

    if self.denoiser_method == 'U-Net':
        
        self.dw = unet.UNet(self.unet_options)                                               
    
    elif self.denoiser_method == 'resnet':
        
        if self.use_shared_weights == True:
            self.dw = dw(self.resnet_options)
        else:
            self.dw = nn.ModuleList([dw(self.resnet_options) for _ in range(self.K)])

class Aclass:
    """
    This class is created to do the data-consistency (DC) step as described in paper.
    """
    def __init__(self, kw_dictionary):
        '''
        Initializes Conjugate gradients step.
        Params:
            - kw_dictionary (dict): Keyword dictionary
        '''
        
        self.img_size = kw_dictionary['image_size']
        self.number_projections = kw_dictionary['number_projections']
        self.lam = kw_dictionary['lambda']
        self.use_torch_radon = kw_dictionary['use_torch_radon']
        self.angles = np.linspace(0, 2*np.pi, self.number_projections,endpoint = False)
        self.det_count = int(np.ceil(np.sqrt(2)*self.img_size))
        
        if self.use_torch_radon == True:
            self.radon = thrad(self.img_size, self.angles, clip_to_circle = False, det_count = self.det_count)
        else:
            class Radon:
                def __init__(self, num_angles, circle=True):
                    self.num_angles = num_angles
                    self.circle = circle
                
                def forward(self, image):
                    # Compute the Radon transform of the image
                    image = image.detach().cpu().numpy()
                    sinogram = radon(image, theta=np.linspace(0, 180, self.num_angles), circle=self.circle)
                    sinogram = torch.tensor(sinogram)
                    return sinogram
                
                def backprojection(self, sinogram):
                    # Compute the backprojection of the sinogram
                    sinogram = sinogram.detach().cpu().numpy()
                    reconstruction = iradon(sinogram, theta=np.linspace(0, 2*180, self.num_angles), circle=self.circle, filter_name=None)
                    reconstruction = torch.tensor(reconstruction)
                    return reconstruction
            
            self.radon = Radon(self.number_projections, circle=False)

    def forward(self, img):
        """
        Applies the operator (A^H A + lam*I) to image, where A is the forward Radon transform.
        Params:
            - img (torch.Tensor): Input tensor
        """

        sinogram = self.radon.forward(img)/self.img_size 
        iradon = self.radon.backprojection(sinogram)*np.pi/self.number_projections
        del sinogram
        output = iradon+self.lam*img
        # print('output forward: {} {}'.format(output.max(), output.min()))
        # print('Term z max {}, min {}'.format((iradon/self.lam).max(), (iradon/self.lam).min()))
        # print('Term input max {}, min {}'.format(img.max(), img.min()))
        # print('Term output max {}, min {}'.format(output.max(), output.min()))
        return output
    
    def inverse(self, rhs):
        """
        Applies CG on each image on the batch
        Params: 
            - rhs (torch.Tensor): Right-hand side tensor for applying inversion of (A^H A + lam*I) operator
        """

        y = torch.zeros_like(rhs)

        for i in range(rhs.shape[0]):
                
            y[i,0,:,:] = self.conjugate_gradients(self.forward, rhs[i,0,:,:]) # This indexing may fail
        
        return y
    
    @staticmethod
    def conjugate_gradients(A, rhs):
        
        """
        My implementation of conjugate gradients in PyTorch
        """

        i = 0
        x = torch.zeros_like(rhs)
        r = rhs 
        p = rhs 
        rTr = torch.sum(r*r)
        
        while((i<10) and torch.ge(rTr, 1e-5)):
            
            Ap = A(p)
            alpha = rTr/torch.sum(p*Ap)
            x = x + alpha*p
            r = r - alpha*Ap
            rTrNew = torch.sum(r*r)
            beta = rTrNew/rTr
            p = r + beta * p
            i += 1
            rTr = rTrNew

        # print('output CG: {} {}'.format(x.max(), x.min()))
        return x

def normalize_images(images):
    '''
    Normalizes tensor of images 1-channel images between 0 and 1.
    Params:
     - images (torch.Tensor): Tensor of 1-channel images
    '''
    
    image_norm = torch.zeros_like(images)


    for i, image in enumerate(images):

        # print(image.max())
        image = (image-image.mean())/image.std()
        image_norm[i,...] = ((image - image.min())/(image.max()-image.min()))

    return image_norm
