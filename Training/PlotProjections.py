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

train_name_modl = 'Optimization_Projections_PSNR_MODL_Test62'

with open(results_folder+train_name_modl+'Projections_SSIM_PSNR.pkl', 'rb') as f:
        
    test_loss_dict = pickle.load(f)

fig, ax= plt.subplots(1,2, figsize = (12,6))
col = ['red', 'blue', 'green']

for proj, losses in test_loss_dict.items():
    
    for i, (loss_key, val_loss) in enumerate(losses.items()):
        
        val_loss = np.array(val_loss)
        if 'mse' in loss_key:
            
            ax[0].scatter(max_angle//int(proj), val_loss.mean(), c = col[i], label = loss_key)
        
        else:
            ax[1].scatter(max_angle//int(proj), val_loss.mean(), c = col[i//2], label = loss_key)


ax[0].set_xlabel('Acceleration factor')
ax[1].set_xlabel('Acceleration factor')

ax[0].set_ylabel('PSNR in testing images')
ax[1].set_ylabel('SSIM in testing images')
ax[0].legend()
ax[0].grid(True)

fig.savefig(results_folder+'Projections_Comparison_SSIM_PSNR.pdf', bbox_inches = 'tight')
