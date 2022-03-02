"""
Test ADMM and comparison
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#%% ADMM test on data
proj_num = 72
augment_factor = 1
total_size = 5000
n_angles = 72
img_size = 100
det_count = int((img_size+0.5)*np.sqrt(2))
tv_iters = 3

tensor_path_X = datasets_folder + 'Proj_{}_augmentFactor_{}_totalSize_{}_FullX.pt'.format(proj_num, augment_factor, total_size)                                            
tensor_path_Y = datasets_folder + 'Proj_{}_augmentFactor_{}_totalSize_{}_FullY.pt'.format(proj_num, augment_factor, total_size)                                            
tensor_path_FiltX = datasets_folder + 'Proj_{}_augmentFactor_{}_totalSize_{}_FiltFullX.pt'.format(proj_num, augment_factor, total_size)                                            

fullX = torch.load(tensor_path_X, map_location=torch.device('cpu'))
fullY = torch.load(tensor_path_Y, map_location=torch.device('cpu'))
fullFiltX = torch.load(tensor_path_FiltX, map_location=torch.device('cpu'))

# Radon operator
angles = np.linspace(0, 2*180, n_angles, endpoint = False)

Psi = lambda x,th:  RecTV.TVdenoise(x,2/th,tv_iters)
#  set the penalty function, to compute the objective
Phi = lambda x: RecTV.TVnorm(x)
hR = lambda x: radon(x, angles, circle = False)
hRT = lambda sino: iradon(sino, angles, circle = False)

loss_test_ADMM = []

fig_ADMM, ax_ADMM = plt.subplots((fullX.shape[0]//500)//3, 3)
fig_FBP, ax_FBP = plt.subplots((fullX.shape[0]//500)//3, 3)

ax_ADMM = ax_ADMM.flatten()
ax_FBP = ax_FBP.flatten()

for a_ADMM, a_FBP in zip(ax_ADMM, ax_FBP):

    a_ADMM.set_axis_off()
    a_FBP.set_axis_off()

for i, (imageX_test, imageY_test, imageFiltX_test) in enumerate(zip(fullX, fullX, fullFiltX)):
    
    imageY_test = imageY_test[0,...].to(device).cpu().numpy().T
    imageX_test = imageX_test[0,...].to(device).cpu().numpy().T 
    imageFiltX_test = imageFiltX_test[0,...].to(device).cpu().numpy().T

    sino = hR(imageY_test)
    img_rec_FBP = hRT(sino) 
    img_rec_ADMM,_,_,_ = RecTV.ADMM(y = sino, A = hR, AT = hRT, Den = Psi, alpha = 10000, delta = 1, max_iter = 200, phi = Phi, tol = 10e-7, invert = 0, warm = 0, true_img = imageY_test)
    img_rec_ADMM = (img_rec_ADMM-img_rec_ADMM.min())/(img_rec_ADMM.max()-img_rec_ADMM.min())

    mse = ((imageFiltX_test - img_rec_ADMM)**2).sum()
    psnr = round(modutils.psnr(img_size, mse, 1), 3)
    loss_test_ADMM.append(psnr)
    
    if i%500 == 0:
        
        im1 = ax_ADMM[i//500].imshow(img_rec_ADMM)
        im2 = ax_FBP[i//500].imshow(imageFiltX_test)
        

        divider_ADMM = make_axes_locatable(ax_ADMM[i//500])
        divider_FBP = make_axes_locatable(ax_FBP[i//500])
        
        cax_ADMM = divider_ADMM.append_axes("right", size="5%", pad=0.05) 
        cax_FBP = divider_FBP.append_axes("right", size="5%", pad=0.05)

        plt.colorbar(im1, cax=cax_ADMM)
        plt.colorbar(im2, cax=cax_FBP)       
        
        ax_ADMM[i//500].set_title('PSNR = {} dB'.format(psnr))
        ax_FBP[i//500].set_title('PSNR = {} dB'.format(psnr))
        
        break
print(np.array(loss_test_ADMM).mean())
np.savetxt(results_folder+'TestADMM_Results.txt', np.array(loss_test_ADMM))

fig_ADMM.savefig(results_folder+'ADMMReconstructions.pdf', bbox_inches = 'tight')
fig_FBP.savefig(results_folder+'FBPReconstructions.pdf', bbox_inches = 'tight')
