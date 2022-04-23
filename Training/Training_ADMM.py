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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#%% ADMM test on data
K  = 10 
proj_num = 72
lam = 0.05
nLayer = 8
augment_factor = 1
total_size = 5000
n_angles = 72
max_angle = 640
img_size = 100
det_count = int((img_size+0.5)*np.sqrt(2))
tv_iters = 3
lr = 0.001
shrink = 0.5

train_name_MODL = 'Optimization_Layers_CorrectRegistration_Test52'
train_name_MODLUNET = 'UnetModl_NonRes_Test59'
train_name_UNET = 'Unet_ResVersion_Test58'

unet_options = {'residual': False, 'up_conv' :True, 'batch_norm' :True, 'batch_norm_inconv' : True}


model_MODLUNET = modl.OPTmodl(nLayer, K, max_angle, proj_num, img_size, None, lam, True, results_folder, useUnet = 'unet', unet_options = unet_options)
model_UNET = modl.UNet(1,1, residual = True, up_conv = True, batch_norm = True, batch_norm_inconv = True).to(device)
model_MODL = modl.OPTmodl(nLayer, K, max_angle, proj_num, img_size, None, lam, True, results_folder, useUnet = False)

modutils.load_net(model_folder+train_name_MODL+'K_{}_lam_{}_nlay_{}_proj_{}'.format(K, lam, nLayer, proj_num), model_MODL, device)
modutils.load_net(model_folder+train_name_MODLUNET+'Model_ModlUNet_lr{}_shrink{}'.format(lr, shrink), model_MODLUNET, device)
modutils.load_net(model_folder+train_name_UNET+'Model_Unet_lr{}_shrink{}'.format(lr, shrink), model_UNET, device)

tensor_path_X = datasets_folder + 'Proj_{}_augmentFactor_{}_totalSize_{}_FullX.pt'.format(proj_num, augment_factor, total_size)                                            
tensor_path_Y = datasets_folder + 'Proj_{}_augmentFactor_{}_totalSize_{}_FullY.pt'.format(proj_num, augment_factor, total_size)                                            
tensor_path_FiltX = datasets_folder + 'Proj_{}_augmentFactor_{}_totalSize_{}_FiltFullX.pt'.format(proj_num, augment_factor, total_size)                                            

fullX = torch.load(tensor_path_X, map_location=torch.device('cpu'))
fullY = torch.load(tensor_path_Y, map_location=torch.device('cpu'))
fullFiltX = torch.load(tensor_path_FiltX, map_location=torch.device('cpu'))

# Radon operator
angles = np.linspace(0, 2*180, n_angles, endpoint = False)

Psi = lambda x,th: RecTV.TVdenoise(x,2/th,tv_iters)
#  set the penalty function, to compute the objective
Phi = lambda x: RecTV.TVnorm(x)
hR = lambda x: radon(x, angles, circle = False)
hRT = lambda sino: iradon(sino, angles, circle = False)
#Twist parameters
kwargs = {'PSI': Psi, 'PHI':Phi, 'LAMBDA':1e-4, 'TOLERANCEA':1e-4, 'STOPCRITERION': 1, 'VERBOSE': 1, 'INITIALIZATION': 0, 'MAXITERA':10000, 'GPU' : 0}

loss_test_ADMM = []
loss_test_fbp = []
loss_test_modl = []
loss_test_TWIST = []
loss_test_MODLUNET = []
loss_test_UNET = []

fig_ADMM, ax_ADMM = plt.subplots((fullX.shape[0]//500)//3+1, 3, figsize = (20, 20))
fig_FBP, ax_FBP = plt.subplots((fullX.shape[0]//500)//3+1, 3, figsize = (20, 20))
fig_MODL, ax_MODL = plt.subplots((fullX.shape[0]//500)//3+1, 3, figsize = (20, 20))
fig_TWIST, ax_TWIST = plt.subplots((fullX.shape[0]//500)//3+1, 3, figsize = (20,20))
fig_MODLUNET, ax_MODLUNET = plt.subplots((fullX.shape[0]//500)//3+1, 3, figsize = (20,20))
fig_UNET, ax_UNET = plt.subplots((fullX.shape[0]//500)//3+1, 3, figsize = (20,20))

ax_ADMM = ax_ADMM.flatten()
ax_FBP = ax_FBP.flatten()
ax_MODL = ax_MODL.flatten()
ax_TWIST = ax_TWIST.flatten()
ax_MODLUNET = ax_MODLUNET.flatten()
ax_UNET = ax_UNET.flatten()

for a_ADMM, a_FBP, a_MODL, a_TWIST, a_MODLUNET, a_UNET in zip(ax_ADMM, ax_FBP, ax_MODL, ax_TWIST, ax_MODLUNET, ax_UNET):

    a_ADMM.set_axis_off()
    a_FBP.set_axis_off()
    a_MODL.set_axis_off()
    a_TWIST.set_axis_off()
    a_MODLUNET.set_axis_off()
    a_UNET.set_axis_off()

for i, (imageX_test, imageY_test, imageFiltX_test) in enumerate(zip(fullX, fullY, fullFiltX)):
     
    if i%500 == 0:
    
        image_rec_MODL = model_MODL(imageX_test[None,...].to(device))['dc'+str(K)][0,0,...].detach().cpu().numpy().T 
        image_rec_MODLUNET = model_MODLUNET(imageX_test[None,...].to(device))['dc'+str(K)][0,0,...].detach().cpu().numpy().T
        image_rec_UNET = model_UNET(imageFiltX_test[None,...].to(device))[0,0,...].detach().cpu().numpy().T
        image_rec_UNET = (image_rec_UNET-image_rec_UNET.min())/(image_rec_UNET.max()-image_rec_UNET.min())

        imageY_test = imageY_test[0,...].to(device).cpu().numpy().T
        imageX_test = imageX_test[0,...].to(device).cpu().numpy().T 
        imageFiltX_test = imageFiltX_test[0,...].to(device).cpu().numpy().T

        sino = hR(imageFiltX_test)
        img_rec_ADMM,_,_,_ = RecTV.ADMM(y = sino, A = hR, AT = hRT, Den = Psi, alpha = 0.01,delta = 0.5, max_iter = 20, phi = Phi, tol = 10e-7, invert = 0, warm = 1, true_img = imageY_test)
        img_rec_ADMM = (img_rec_ADMM-img_rec_ADMM.min())/(img_rec_ADMM.max()-img_rec_ADMM.min())
        
        print('Image Y test', imageY_test.min(), imageY_test.max()) 
        print('Image ', img_rec_ADMM.min(), img_rec_ADMM.max())

        img_rec_TWIST,_,_,_ = RecTV.TwIST(y = sino, A =hR, AT = hRT, tau = 0.01, kwarg = kwargs , true_img = imageY_test)
       
        mse_admm = ((imageY_test - img_rec_ADMM)**2).sum()
        psnr_admm = round(modutils.psnr(img_size, mse_admm, 1), 3)
        print(psnr_admm)
        loss_test_ADMM.append(psnr_admm)

        mse_twist = ((imageY_test - img_rec_TWIST)**2).sum()
        psnr_twist = round(modutils.psnr(img_size, mse_twist, 1), 3)
        print(psnr_twist)
        loss_test_TWIST.append(psnr_twist)

        mse_fbp = ((imageFiltX_test - imageY_test)**2).sum() 
        psnr_fbp = round(modutils.psnr(img_size, mse_fbp, 1), 3) 
        loss_test_fbp.append(psnr_fbp)

        mse_modl = ((image_rec_MODL - imageY_test)**2).sum() 
        psnr_modl = round(modutils.psnr(img_size, mse_modl, 1), 3) 
        loss_test_modl.append(psnr_modl)
        
        mse_MODLUNET = ((image_rec_MODLUNET - imageY_test)**2).sum() 
        psnr_MODLUNET = round(modutils.psnr(img_size, mse_MODLUNET, 1), 3) 
        loss_test_MODLUNET.append(psnr_MODLUNET)

        mse_UNET = ((image_rec_UNET - imageY_test)**2).sum() 
        psnr_UNET = round(modutils.psnr(img_size, mse_UNET, 1), 3) 
        loss_test_UNET.append(psnr_UNET)
        
        im1 = ax_ADMM[i//500].imshow(img_rec_ADMM)
        im2 = ax_FBP[i//500].imshow(imageY_test)
        im3 = ax_MODL[i//500].imshow(image_rec_MODL)
        im4 = ax_TWIST[i//500].imshow(img_rec_TWIST)
        im5 = ax_MODLUNET[i//500].imshow(image_rec_MODLUNET)
        im6 = ax_UNET[i//500].imshow(image_rec_UNET)

        divider_ADMM = make_axes_locatable(ax_ADMM[i//500])
        divider_FBP = make_axes_locatable(ax_FBP[i//500]) 
        divider_MODL = make_axes_locatable(ax_MODL[i//500])
        divider_TWIST = make_axes_locatable(ax_TWIST[i//500])
        divider_MODLUNET = make_axes_locatable(ax_MODLUNET[i//500])
        divider_UNET = make_axes_locatable(ax_UNET[i//500])

        cax_ADMM = divider_ADMM.append_axes("right", size="5%", pad=0.05) 
        cax_FBP = divider_FBP.append_axes("right", size="5%", pad=0.05)
        cax_MODL = divider_MODL.append_axes("right", size="5%", pad=0.05)
        cax_TWIST = divider_TWIST.append_axes("right", size="5%", pad=0.05)
        cax_MODLUNET = divider_MODLUNET.append_axes("right", size="5%", pad=0.05)
        cax_UNET = divider_UNET.append_axes("right", size="5%", pad=0.05)
        
        plt.colorbar(im1, cax=cax_ADMM)
        plt.colorbar(im2, cax=cax_FBP)       
        plt.colorbar(im3, cax=cax_MODL)
        plt.colorbar(im4, cax=cax_TWIST) 
        plt.colorbar(im5, cax=cax_MODLUNET)
        plt.colorbar(im6, cax=cax_UNET)

        ax_ADMM[i//500].set_title('PSNR = {} dB'.format(psnr_admm))
        ax_FBP[i//500].set_title('PSNR = {} dB'.format(psnr_fbp)) 
        ax_MODL[i//500].set_title('PSNR = {} dB'.format(psnr_modl))
        ax_TWIST[i//500].set_title('PSNR = {} dB'.format(psnr_twist))
        ax_MODLUNET[i//500].set_title('PSNR = {} dB'.format(psnr_MODLUNET))
        ax_UNET[i//500].set_title('PSNR = {} dB'.format(psnr_UNET))

        fig_ADMM.savefig(results_folder+'ADMMReconstructions.pdf', bbox_inches = 'tight')
        fig_FBP.savefig(results_folder+'FBPReconstructions.pdf', bbox_inches = 'tight')
        fig_MODL.savefig(results_folder+'MODLReconstructions.pdf', bbox_inches = 'tight')
        fig_TWIST.savefig(results_folder+'TWISTReconstructions.pdf', bbox_inches = 'tight')
        fig_MODLUNET.savefig(results_folder+'MODLUNETReconstructions.pdf', bbox_inches = 'tight')
        fig_UNET.savefig(results_folder+'UNETReconstructions.pdf', bbox_inches = 'tight')
        

print(np.array(loss_test_ADMM).mean())
print(np.array(loss_test_TWIST).mean())
print(np.array(loss_test_modl).mean())
print(np.array(loss_test_fbp).mean())

loss_tests = {'ADMM': loss_test_ADMM, 'MODL': loss_test_modl, 'FBP' : loss_test_fbp, 'TWIST': loss_test_TWIST, 'MODLUNET': loss_test_MODLUNET, 'UNET' : loss_test_UNET}

with open(results_folder+'TestADMM_Results.txt', 'wb') as f:

    pickle.dump(loss_tests, f)


# Load UNet Results



