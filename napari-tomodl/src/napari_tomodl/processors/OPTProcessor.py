
from skimage.transform import radon, iradon
import numpy as np
# from torch_radon import Radon, RadonFanbeam
# from torch_radon.solvers import cg
import matplotlib.pyplot as plt 


class OPTProcessor:

    def __init__(self):

        pass
    
    
    def generate_angles(self):
        '''
        Generate according angles for reconstruction
        '''

        self.angles = np.linspace(0., 180., 100, endpoint=False)

    
    
