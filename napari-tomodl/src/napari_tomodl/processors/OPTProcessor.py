'''
Process sinograms in 2D
'''

from skimage.transform import radon, iradon
import numpy as np
from napari.layers import Image
# from torch_radon import Radon, RadonFanbeam
# from torch_radon.solvers import cg
import os
import matplotlib.pyplot as plt 
import cv2

for k, v in os.environ.items():
    if k.startswith("QT_") and "cv2" in v:
        del os.environ[k]



class OPTProcessor:

    def __init__(self):
        '''
        Variables for OPT processor
        '''

        self.resizeVal = 100
        self.recProcess = 'FBP (CPU)'
        
        self.resizeBool = True
        self.registerBool = False
    
    def reshape_input(self, sinogram):
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

    def correct_sinogram(self, sinogram):
        '''
        Corrects axis of rotation for artifact removal
        '''
        pass

    def reconstruct(self, sinogram):
        '''
        Reconstruct with specific method
        '''
        if self.resizeBool == True:
            print(sinogram.shape)
            sinogram_resize = cv2.resize(sinogram, (sinogram.shape[0], int((self.resizeVal+0.5)*np.sqrt(2))), interpolation = cv2.INTER_AREA)
            print(sinogram_resize.shape)

        angles = np.linspace(0, 2*180, sinogram_resize.shape[1], endpoint = False)

        return iradon(sinogram_resize, angles, circle = False)
        
    
