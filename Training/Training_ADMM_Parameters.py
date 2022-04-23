"""
Search for optimal parameters alpha and delta in ADMM
"""
#%% Import libraries
import os
import os,time, sys
os.chdir('.')
sys.path.append('Utilities/')
sys.path.append('OPTmodl/')
sys.path.append('Reconstruction/')

import torch.nn.functional as F
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

proj_num = 72
augment_factor = 1
total_size = 5000
tv_iters = 3
n_angles = 72

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
#Twist parameters
kwargs = {'PSI': Psi, 'PHI':Phi, 'LAMBDA':1e-4, 'TOLERANCEA':1e-4, 'STOPCRITERION': 1, 'VERBOSE': 1, 'INITIALIZATION': 0, 'MAXITERA':10000, 'GPU' : 0}

fig_MSE, ax_MSE = plt.subplots(1, 1, figsize = (10, 10))
fig_SSIM, ax_SSIM = plt.subplots(1, 1, figsize = (10, 10))


objectives = {}
errors_SSIM = {}
errors_MSE = {}

deltas = np.arange(0.5, 1.5, 0.5)
K = 0

with open(results_folder+'ADMM_Delta_Optimization.pkl', 'rb') as f:

    ADMM_parameters = pickle.load(f)
    dict_load = False

errors_SSIM = ADMM_parameters['SSIM']
errors_MSE = ADMM_parameters['PSNR']

fig_ADMM, ax_ADMM = plt.subplots((fullX.shape[0]//500)//3+1, 3, figsize = (20, 20))

ax_ADMM = ax_ADMM.flatten()

for a_ADMM in ax_ADMM:
    a_ADMM.set_axis_off()

for delta in deltas:
    
    if dict_load != True:
        for i, (imageX_test, imageY_test, imageFiltX_test) in enumerate(zip(fullX, fullY, fullFiltX)):
            
            if i%500 == 0:

                imageY_test = imageY_test[0,...].to(device).cpu().numpy().T
                imageX_test = imageX_test[0,...].to(device).cpu().numpy().T 
                imageFiltX_test = imageFiltX_test[0,...].to(device).cpu().numpy().T

                K +=1 #contador de imagenes 

                sino = hR(imageFiltX_test)

                img_rec_ADMM, objective, error_MSE, error_SSIM = RecTV.ADMM(y = sino, A = hR, AT = hRT, Den = Psi, alpha = 0.05, delta = delta, max_iter = 20, phi = Phi, tol = 10e-7, invert = 0, warm = 1, true_img = imageY_test)    
                if delta == 0.5:
                    im1 = ax_ADMM[i//500].imshow(img_rec_ADMM)
                    divider_ADMM = make_axes_locatable(ax_ADMM[i//500])
                    cax_ADMM = divider_ADMM.append_axes("right", size="5%", pad=0.05) 
                    plt.colorbar(im1, cax=cax_ADMM)

                if i == 0:

                    objectives[delta] = np.array(objective)
                    errors_SSIM[delta] = np.array(error_SSIM)
                    errors_MSE[delta] = 10*np.log10(1/error_MSE)

                # Paso a PSNR?
                objectives[delta] += np.array(objective)
                errors_SSIM[delta] += np.array(error_SSIM)
                errors_MSE[delta] += 10*np.log10(1/error_MSE)
                
                if delta == 0.5:
                    ax_ADMM[i//500].set_title('PSNR = {} dB'.format(round(10*np.log10(1/error_MSE[-1]), 3)))
                print('error calculado aparte:', 10*np.log10(1/((img_rec_ADMM-imageY_test)**2).mean()))
                print('Error de programa:', errors_MSE[delta][-1])
        
        errors_SSIM[delta] /= K
        objectives[delta] /= K
        errors_MSE[delta] /= K
    
        ax_SSIM.plot(errors_SSIM[delta], label = delta)
        ax_MSE.plot(errors_MSE[delta], label = delta)
    
    else:
        ax_SSIM.plot(errors_SSIM[delta], label = delta)
        ax_MSE.plot(errors_MSE[delta], label = delta)
    
ax_SSIM.set_xlabel('Iterations')
ax_SSIM.set_ylabel('SSIM')

ax_MSE.set_ylabel('PSNR [dB]')
ax_MSE.set_xlabel('Iterations')

ax_MSE.grid(True)
ax_SSIM.grid(True)

ax_MSE.legend()
ax_SSIM.legend()

fig_ADMM.savefig(results_folder+'ADMMReconstructions.pdf', bbox_inches = 'tight')
fig_SSIM.savefig(results_folder + 'ADMM_Optimisation_SSIM.pdf', bbox_inches = 'tight')
fig_MSE.savefig(results_folder + 'ADMM_Optimisation_MSE.pdf', bbox_inches = 'tight')

ADMM_Parameters = {'SSIM': errors_SSIM, 'PSNR' : errors_MSE}
ADMM_params_PSNR = errors_MSE[0.5][-1]

with open(results_folder+'TestADMM_Parameters_Results.pkl', 'wb') as f:

    pickle.dump(ADMM_params_PSNR, f)

with open(results_folder+'ADMM_Delta_Optimization.pkl', 'wb') as f:

    pickle.dump(ADMM_Parameters, f)



