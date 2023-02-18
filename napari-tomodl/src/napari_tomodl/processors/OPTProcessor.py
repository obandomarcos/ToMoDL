'''
Process sinograms in 2D
'''

from skimage.transform import radon as radon_scikit 
from skimage.transform import iradon as iradon_scikit

from torch_radon import Radon as radon_thrad

import torch
import numpy as np
from napari.layers import Image
import scipy.ndimage as ndi
# from torch_radon import Radon, RadonFanbeam
# from torch_radon.solvers import cg
import os
import matplotlib.pyplot as plt 
import cv2
from enum  import Enum

for k, v in os.environ.items():
    if k.startswith("QT_") and "cv2" in v:
        del os.environ[k]

class Rec_modes(Enum):
    FBP_CPU = 0
    FBP_GPU = 1

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

        print(self.rec_process)
        # This should change depending on the method
        if self.rec_process == Rec_modes.FBP_CPU.value:
            
            self.angles_gen = lambda num_angles: np.linspace(0, 2*180, num_angles, endpoint = False)
            self.iradon_function = lambda sino, num_angles: iradon_scikit(sino, self.angles_gen(num_angles), circle = False)
        
        elif self.rec_process == Rec_modes.FBP_GPU.value:

            self.angles_gen = lambda num_angles: np.linspace(0, 2*np.pi, num_angles, endpoint = False)

            self.iradon_functor = lambda num_angles: radon_thrad(self.resize_val, self.angles_gen(num_angles))
            print(self.angles)

            self.iradon_function = lambda sino, num_angles: self.iradon_functor(num_angles).backward(self.iradon_functor(num_angles).filter_sinogram(torch.tensor(sino))).numpy()


    def reshape_input(self, sinogram:np.ndarray):
        '''
        Reshape stack volume for output visualization
        '''
        self.volume = self.layer.data

        for layer in self.viewer.layers:
            if isinstance(layer, Image):
                if layer.ndim >2:
                    scale = layer.scale 
                    scale[-3] = self.zscaling
                    layer.scale = scale

    def correct_and_reconstruct(self, sinogram: np.ndarray):
        '''
        Corrects rotation axis by finding optimal registration via maximising reconstructed image's intensity variance.

        Based on 'Walls, J. R., Sled, J. G., Sharpe, J., & Henkelman, R. M. (2005). Correction of artefacts in optical projection tomography. Physics in Medicine & Biology, 50(19), 4645.'

        Params:
        '''

        self.angles = self.angles_gen(sinogram.shape[0])

        shifts = np.arange(-self.max_shift, self.max_shift, self.shift_step)+self.center_shift
        image_std = []

        for i, shift in enumerate(shifts):

            sino_shift = ndi.shift(sinogram, (shift, 0), mode = 'nearest')

            # Get image reconstruction
            shift_iradon = self.reconstruct(sino_shift)
            
            # Calculate variance
            image_std.append(np.std(shift_iradon))
        
        # To-Do: Change shifts
        self.center_shift = shifts[np.argmax(image_std)]
        self.max_shift = 5
        self.shift_step = 0.5

    
        return self.reconstruct(ndi.shift(sinogram, (self.center_shift, 0), mode = 'nearest'))


    def reconstruct(self, sinogram: np.ndarray):
        '''
        Reconstruct with specific method
        TODO: Include other methods
        '''
        if self.resize_bool == True:
            
            sinogram_resize = cv2.resize(sinogram, (sinogram.shape[0], int(np.ceil(self.resize_val*np.sqrt(2)))), interpolation = cv2.INTER_AREA)

        return self.iradon_function(sinogram_resize, sinogram.shape[0])
        
    
