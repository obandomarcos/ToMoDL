"""
Test ADMM and comparison
"""
#%% Import libraries
import os
import os,time, sys
os.chdir('/home/obanmarcos/Balseiro/Maestría/Proyecto/Implementación/DeepOPT/')
sys.path.append('Utilities/')
sys.path.append('OPTmodl/')
sys.path.append('Reconstruction/')

import numpy as np
import random
import matplotlib.pyplot as plt
import DataLoading as DL
from Folders_cluster import *
import Reconstruction as RecTV
import ModelUtilities as modutils
import torch
from torch_radon import Radon, RadonFanbeam
from skimage.transform import radon, iradon
import torchvision
import model_torch as modl
import pickle
from tqdm import tqdm
from bayes_opt import BayesianOptimization

#%% ADMM test on data
proj_num = 72
augment_factor = 1
total_size = 5000
n_angles = 72
img_size = 100
det_count = int((img_size+0.5)*np.sqrt(2))

tensor_path = 'Datasets/Proj_{}_augmentFactor_{}_totalSize_{}_FullY.pt'.format(proj_num, augment_factor, total_size)                                            

fullY = torch.load(tensor_path, map_location=torch.device('cpu'))

#%%

# Tensors in image space
# dataloaders = modutils.formDataloaders(datasets, proj_num, total_size, train_factor, val_factor, test_factor, batch_size, img_size, tensor_path, augment_factor, load_tensor = True, save_tensor = False)    

# Radon operator
angles = np.linspace(0, 2*np.pi, n_angles, endpoint = False)
    
radon = Radon(img_size, angles, clip_to_circle = False, det_count = det_count)

# Have to send FiltX to Sinogram space in order to use ADMM
plt.imshow(radon.forward(fullY[0, 0, ...]))







