"""
Compare Test Results ADMM/FBP/ResMODL/UnetMODL/Unet/
"""
#%% Import libraries
import os
import os,time, sys
os.chdir('.')
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
from mpl_toolkits.axes_grid1 import make_axes_locatable

train_name_unet = 'Unet_ResVersion_Test58'
train_name_modlunet = 'UnetModl_NonRes_Test59'

lr, shrink = 0.001, 0.5

#%% Import all files
# ResMoDL, ADMM, TwIST
with open(results_folder+'TestADMM_Results.txt', 'rb') as f:
    loss_test = pickle.load(f)


with open(results_folder+'TestADMM_Parameters_Results.pkl', 'rb') as f:

    ADMM_params = pickle.load(f)

# UnetMoDL, Unet
with open(results_folder+train_name_modlunet+'Test_UnetMoDL_lr{}_shrink{}.pkl'.format(lr, shrink), 'rb') as f:
    
    unpickle = pickle.Unpickler(f) 
    test_loss_modlUnet = unpickle.load()
    print('Diccionario cargado para proyección {}, MODL+UNET')

with open(results_folder+train_name_unet+'Test_Unet_lr{}_shrink{}.pkl'.format(lr, shrink), 'rb') as f:
    
    unpickle = pickle.Unpickler(f)
    test_loss_Unet = unpickle.load()
    print('Diccionario cargado para UNET')

loss_test['ResMoDL'] = loss_test.pop('MODL') 
loss_test['MODLUNET'] = test_loss_modlUnet['loss_net'] 
loss_test['UNET'] = test_loss_Unet['loss_net']
loss_test['ADMM'] = ADMM_params

fig, ax = plt.subplots(1,1, figsize = (6,6))

for (key, value), i in zip(loss_test.items(), range(len(list(loss_test.keys())))):

    ax.scatter(i, np.array(value).mean())

ax.set_xticks(np.arange(len(loss_test.values())))
ax.set_xticklabels(loss_test.keys())
ax.grid(True)
ax.set_ylabel('PSNR [dB]')

fig.savefig(results_folder+'AllMethodsComparison.pdf', bbox_inches = 'tight')




