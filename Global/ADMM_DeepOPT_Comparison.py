'''
Comparison of implemented methods on OPT data
'''
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
#%%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
folder_paths = [ f140315_3dpf] # Folders to be used

umbral_reg = 50
proj_num = 72

train_factor = 0.7
val_factor = 0.2
test_factor = 0.1
total_size = 3000

batch_size = 5
img_size = 100
augment_factor = 10
train_infos = {}
test_loss_dict = {}
# tensor_path = datasets_folder+'Proj_{}_augmentFactor_{}_totalSize_{}_'.format(proj_num, augment_factor, total_size)

datasets = modutils.formRegDatasets(folder_paths, umbral_reg, img_resize = img_size)

#%% ADMM TV Parameters
tv_iters = 3
volume = datasets[0]
beams = 90

print(volume.shape)
# Angles
theta = np.linspace(0., 180.-180/beams, beams)
beams_projections = np.linspace(0, volume.shape[0], beams).astype(int)
subsampled_volume = volume[beams_projections,...]

Psi = lambda x,th:  RecTV.TVdenoise(x,2/th,tv_iters)
#  set the penalty function, to compute the objective
Phi = lambda x: RecTV.TVnorm(x)
hR = lambda x: radon(x, theta, circle = False)
hRT = lambda sino: iradon(sino, theta, circle = False)



