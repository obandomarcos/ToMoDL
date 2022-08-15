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
train_name_SSIM = 'Optimization_K_SSIM_MSE_Test65'

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

proj_num = 72
K = 7
nLayer = 8
train_factor = 0.7
lam = 0.001

with open(results_folder+train_name_SSIM+'KAnalisis_Proj{}_nLay{}_K{}_lam{}_trnSize{}.pkl'.format(proj_num, nLayer, K, lam, train_factor), 'rb') as f:
    
    unpickle = pickle.Unpickler(f)
    test_loss_SSIM = unpickle.load()
    print('Diccionario cargado para SSIM trained netTr')


print(test_loss_SSIM[7].keys())

loss_test['ResMoDL-SSIM'] = test_loss_SSIM[K]['loss_mse']
loss_test['ResMoDL'] = loss_test.pop('MODL') 
loss_test['MODLUNET'] = test_loss_modlUnet['loss_net'] 
loss_test['UNET'] = test_loss_Unet['loss_net']
loss_test['ADMM'] = ADMM_params

fig, ax = plt.subplots(1,1, figsize = (6,6))

for (key, value), i in zip(loss_test.items(), range(len(list(loss_test.keys())))):
    print(key, np.array(value).mean(), print(np.array(value).shape))
    ax.scatter(i, np.array(value).mean())#, yerr = np.array(value).std())

ax.set_xticks(np.arange(len(loss_test.values())))
ax.set_xticklabels(loss_test.keys())
ax.grid(True)
ax.set_ylabel('PSNR [dB]')

fig.savefig(results_folder+'AllMethodsComparison.pdf', bbox_inches = 'tight')




