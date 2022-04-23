"""
Train the model with different projections angles
author: obanmarcos
"""
import os
import os,time, sys
os.chdir('/home/marcos/DeepOPT/')
sys.path.append('Utilities/')
sys.path.append('OPTmodl/')
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import DataLoading as DL
from Folders_cluster import *
import ModelUtilities as modutils
import torch
from torch_radon import Radon, RadonFanbeam
from skimage.transform import radon, iradon
import phantominator as ph
import torchvision
import model_torch as modl
import pickle
from tqdm import tqdm

number_projections = 720//np.arange(2, 26, 2)
factors = np.arange(2,26,2)

train_factor = 0.7
val_factor = 0.2
test_factor = 0.1 
total_size = 5000                  
batch_size = 5 
img_size = 100
augment_factor = 1

#%% Model Settings
shrink = 0.5
nLayer = 2
K = 8
epochs = 1
lam = 0.05
max_angle = 720
lr = 0.001
proj_num = 45

train_name = 'Optimization_Projections_MoDL_Test60_'
train_name_Unet = 'Optimization_Projections_UnetResidual_Test61_'

with open(results_folder+train_name+'Unique_Proj{}_nLay{}_epochs{}_K{}_lam{}_trnSize{}.pkl'.format(proj_num, nLayer, epochs, K, lam, train_factor), 'rb') as f:
        
    test_loss_dict = pickle.load(f)
    print('Diccionario salvado para proyección {}'.format(proj_num))

epochs = 50
proj_num = 36

with open(results_folder+train_name+'Unique_Proj{}_nLay{}_epochs{}_K{}_lam{}_trnSize{}.pkl'.format(proj_num, nLayer, epochs, K, lam, train_factor), 'rb') as f:
        
    test_loss_dict_extra = pickle.load(f)
    print('Diccionario salvado para proyección {}'.format(proj_num))


fig, ax_proj = plt.subplots(1,1, figsize = (8,6))

for i, (key_proj, projection_dict) in enumerate(test_loss_dict.items()):
     
    if i == 1:
     
        label_fbp = 'FBP'
        label_modl = 'MoDL'
        label_unet = 'U-Net'
        
    elif i==0:
        continue
   
    else:
        
        label_fbp = label_modl = label_unet = '_nolegend_'

    ax_proj.scatter(720//int(key_proj), np.array(projection_dict['loss_net_modl']).mean(), label = label_modl, color = 'blue')
    ax_proj.scatter(720//int(key_proj), np.array(projection_dict['loss_fbp']).mean(), label = label_fbp, color = 'red')
    ax_proj.scatter(720//int(key_proj), np.array(projection_dict['loss_net_unet']).mean(), label = label_unet, color = 'green')


for i, (key_proj, projection_dict) in enumerate(test_loss_dict_extra.items()):

    if key_proj < 45:

        label_fbp = label_modl = label_unet = '_nolegend_'
                                                                                                                             
        ax_proj.scatter(720//int(key_proj), np.array(projection_dict['loss_net_modl']).mean(), label = label_modl, color = 'blue')
        ax_proj.scatter(720//int(key_proj), np.array(projection_dict['loss_fbp']).mean(), label = label_fbp, color = 'red')
        ax_proj.scatter(720//int(key_proj), np.array(projection_dict['loss_net_unet']).mean(), label = label_unet, color = 'green')

ax_proj.set_xlabel('Acceleration factor')
ax_proj.set_ylabel('PSNR in testing images')
ax_proj.legend()
ax_proj.grid(True)

fig.savefig(results_folder+'Projections_Comparison.pdf', bbox_inches = 'tight')
