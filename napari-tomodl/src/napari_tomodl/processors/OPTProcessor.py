'''
Process sinograms in 2D
'''

from skimage.transform import radon as radon_scikit 
from skimage.transform import iradon as iradon_scikit

from torch_radon import Radon as radon_thrad
from .alternating import TwIST, TVdenoise, TVnorm
from .modl import ToMoDL
import torch
import numpy as np
from napari.layers import Image
import scipy.ndimage as ndi
import os
import matplotlib.pyplot as plt 
import cv2
from enum  import Enum

for k, v in os.environ.items():
    if k.startswith("QT_") and "cv2" in v:
        del os.environ[k]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Rec_modes(Enum):
    FBP_CPU = 0
    FBP_GPU = 1
    TWIST_CPU = 2
    UNET_GPU = 3
    MODL_GPU = 4


class OPTProcessor:

    def __init__(self):
        '''
        Variables for OPT processor
        '''

        self.resize_val = 100
        self.rec_process = Rec_modes.FBP_CPU.value
        
        self.resize_bool = True
        self.register_bool = True
        self.max_shift = 50
        self.shift_step = 10
        self.center_shift = 0

        self.set_reconstruction_process()
    
    def set_reconstruction_process(self):

        self.init_volume_rec = False
        self.iradon_functor = None
        # This should change depending on the method
        if (self.rec_process == Rec_modes.FBP_CPU.value) or (self.rec_process == Rec_modes.MODL_GPU.value):
            
            self.angles_gen = lambda num_angles: np.linspace(0, 2*180, num_angles, endpoint = False)
            self.iradon_function = lambda sino, num_angles: iradon_scikit(sino.T, self.angles_gen(num_angles), circle = False)
        
        elif (self.rec_process == Rec_modes.FBP_GPU.value):
            
            assert(torch.cuda.is_available() == True)
            self.angles_gen = lambda num_angles: np.linspace(0, 2*np.pi, num_angles, endpoint = False)


    def correct_and_reconstruct(self, sinogram: np.ndarray):
        '''
        Corrects rotation axis by finding optimal registration via maximising reconstructed image's intensity variance.

        Based on 'Walls, J. R., Sled, J. G., Sharpe, J., & Henkelman, R. M. (2005). Correction of artefacts in optical projection tomography. Physics in Medicine & Biology, 50(19), 4645.'

        Params:
        '''

        shifts = np.arange(-self.max_shift, self.max_shift, self.shift_step)+self.center_shift
        image_std = []

        print(sinogram.shape)
        for i, shift in enumerate(shifts):

            sino_shift = ndi.shift(sinogram, (0, shift), mode = 'nearest')

            # Get image reconstruction
            shift_iradon = self.reconstruct(sino_shift)
            
            # Calculate variance
            image_std.append(np.std(shift_iradon))
        
        # To-Do: Change shifts
        self.center_shift = shifts[np.argmax(image_std)]
        self.max_shift = 5
        self.shift_step = 0.5

        print(self.max_shift, self.shift_step, self.center_shift)

        return np.fliplr(self.reconstruct(ndi.shift(sinogram, (0, self.center_shift), mode = 'nearest')))

    def resize(self, sinogram_volume: np.ndarray):

        if self.resize_bool == True:
            
            sinogram_resize = np.zeros((sinogram_volume.shape[0], int(np.ceil(self.resize_val*np.sqrt(2))), sinogram_volume.shape[2]), dtype = np.float32)

            for idx in range(sinogram_volume.shape[2]):
                
                sinogram_resize[:,:,idx] = cv2.resize(sinogram_volume[:,:,idx], (sinogram_volume.shape[0], int(np.ceil(self.resize_val*np.sqrt(2)))), interpolation = cv2.INTER_AREA).T

        return sinogram_resize
    
    def reconstruct(self, sinogram: np.ndarray):
        '''
        Reconstruct with specific method
        TODO: 
            * Include methods ToMODL
            * Optimize GPU usage with tensors

        '''

        ## Es un enriedo, pero inicializa los generadores de ángulos. Poco claro
        if self.init_volume_rec == False:

            self.angles = self.angles_gen(sinogram.shape[0])   

        if self.iradon_functor == None:
            
            self.iradon_functor = radon_thrad(self.resize_val, self.angles, clip_to_circle = False, det_count = int(np.ceil(np.sqrt(2)*self.resize_val)))         

        if self.rec_process == Rec_modes.FBP_GPU.value:

            self.iradon_function = lambda sino: self.iradon_functor.backprojection((torch.Tensor(sino).to(device))).cpu().numpy()

        elif self.rec_process == Rec_modes.FBP_CPU.value:
            
            self.iradon_function = lambda sino: iradon_scikit(sino.T, self.angles, circle = False, filter_name = None)
        
        elif self.rec_process == Rec_modes.MODL_GPU.value:
            
            resnet_options_dict = {'number_layers': 5,
                                    'kernel_size':3,
                                    'features':64,
                                    'in_channels':1,
                                    'out_channels':1,
                                    'stride':1, 
                                    'use_batch_norm': True,
                                    'init_method': 'xavier'}
            
            self.tomodl_dictionary = {'use_torch_radon': False,
                                    'metric': 'psnr',
                                    'K_iterations' : 2,
                                    'number_projections_total' : sinogram.shape[0],
                                    'acceleration_factor': 10,
                                    'image_size': 100,
                                    'lambda': 0.00001,
                                    'use_shared_weights': True,
                                    'denoiser_method': 'resnet',
                                    'resnet_options': resnet_options_dict,
                                    'in_channels': 1,
                                    'out_channels': 1}
                            
            self.iradon_functor = ToMoDL(self.tomodl_dictionary)
    
            self.iradon_function = lambda sino: self.iradon_functor(torch.Tensor(iradon_scikit(sino.T, self.angles, circle = False, filter_name = None)).to(device).unsqueeze(0).unsqueeze(1))['dc'+str(self.tomodl_dictionary['K_iterations'])].detach().cpu().numpy()

        elif self.rec_process == Rec_modes.TWIST_CPU.value:

            Psi = lambda x,th: TVdenoise(x,2/th, 3)
            #  set the penalty function, to compute the objective
            Phi = lambda x: TVnorm(x)

            twist_dictionary = {'LAMBDA': 1e-4, 
                                'TOLERANCEA':1e-4,
                                'STOPCRITERION':1, 
                                'VERBOSE':0,
                                'INITIALIZATION':0,
                                'MAXITERA':10000, 
                                'GPU':0,
                                'PSI': Psi,
                                'PHI': Phi,

                                }
            
            A = lambda x: radon_scikit(x, self.angles, circle = False)
            AT = lambda sino: iradon_scikit(sino, self.angles, circle = False)
            self.iradon_function = lambda sino: TwIST(sino.T, A, AT, 0.01, twist_dictionary, true_img = AT(sino.T))[0]
            
        elif self.rec_process == Rec_modes.UNET_GPU.value:    

            pass

            
        reconstruction = self.iradon_function(sinogram)

        return reconstruction
        
    
