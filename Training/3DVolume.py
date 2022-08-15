"""
Reconstruct 3D Volume
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
from pathlib import Path
import cv2 
import matplotlib.patches as patches

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
folder_paths = [f140114_5dpf] # Folders to be used
image_folder = 'f140114_5dpf'

#%% Datasets 
# Training with more than one dataset
# Projnum == %10 of the data
proj_num = 72

train_factor = 0.7
val_factor = 0.2
test_factor = 0.1                 
batch_size= 1 
img_size = 100
augment_factor = 1
train_infos = {}        
total_size = 800 

fish_parts = ['head']

test_loss_dict = {} 

tensor_path = ''
datasets = modutils.formRegDatasets(folder_paths, img_resize =img_size, fish_parts = fish_parts)
dataloaders = modutils.formDataloaders(datasets, proj_num, total_size, train_factor, val_factor, test_factor, batch_size, img_size, tensor_path, augment_factor, load_tensor = False, save_tensor = False, use_rand = True)   

train_name_modl = 'Optimization_Projections_PSNR_MODL_Test62'
train_name_SSIM = 'Optimization_Projections_SSIM_MODL_Test62'

shrink = 0.5
nLayer = 8
K = 8
epochs = 1
lam = 0.05
max_angle = 720
lr = 0.001 

# Load models

#model_MODL = modl.OPTmodl(nLayer, K, max_angle, proj_num, img_size, None, lam, results_folder, shared = True, unet_options = False) 
#modutils.load_net(model_folder+train_name_modl+'K_{}_lam_{}_nlay_{}_proj_{}'.format(K, lam, nLayer, proj_num), model_MODL, device)

# preevisory
train_name_SSIM = 'Optimization_K_SSIM_MSE_Test65K_7_lam_0.001_nlay_8_proj_72'

model_SSIM = modl.OPTmodl(nLayer, K, max_angle, proj_num, img_size, None, lam, results_folder, True, useUnet = False)
modutils.load_net(model_folder+train_name_SSIM, model_SSIM, device)

# Create folders if they don't exist
folder_MODL = volumes_folder+'MODL_PSNR_X{}_K{}_nLayers{}/'.format(max_angle//proj_num, K, nLayer)
folder_SSIM = volumes_folder+'MODL_SSIM_X{}_K{}_nLayers{}/'.format(max_angle//proj_num, K, nLayer)
folder_FBP = volumes_folder+'FBP_X{}/'.format(max_angle//proj_num)
folder_FBP_FULL = volumes_folder+'FBP_X{}/'.format(1)

Path(folder_MODL).mkdir(parents=True, exist_ok=True)
Path(folder_SSIM).mkdir(parents=True, exist_ok=True)
Path(folder_FBP).mkdir(parents=True, exist_ok=True)
Path(folder_FBP_FULL).mkdir(parents=True, exist_ok=True)

z = 0

def render3D():

    for phase in ['train', 'val', 'test']:

        for inp, target, filt in tqdm(zip(dataloaders[phase]['x'], dataloaders[phase]['y'], dataloaders[phase]['filtX'])): 
            
            img_fbp = (255*filt.detach().cpu().numpy())[0,0,...]
            img_modl_PSNR = (255*model_MODL(inp)['dc'+str(K)].detach().cpu().numpy()).astype(int)[0,0,...]
            img_modl_SSIM = (255*model_SSIM(inp)['dc'+str(K)].detach().cpu().numpy()).astype(int)[0,0,...]
            img_fbp_FULL = (255*target.detach().cpu().numpy())[0,0,...]
            
            cv2.imwrite(folder_FBP+image_folder+'_reconstructed_{}.jpg'.format(z), img_fbp) 
            cv2.imwrite(folder_MODL+image_folder+'_reconstructed_{}.jpg'.format(z), img_modl_PSNR)
            cv2.imwrite(folder_SSIM+image_folder+'_reconstructed_{}.jpg'.format(z), img_modl_SSIM)
            cv2.imwrite(folder_FBP_FULL+image_folder+'_reconstructed_{}.jpg'.format(z), img_fbp_FULL)
            z += 1

def denoise_stages():
    
    phase = 'test'
    for i, (inp, target, filt) in enumerate(zip(dataloaders[phase]['x'], dataloaders[phase]['y'], dataloaders[phase]['filtX'])): 
        
        if i == 100:
            inp = inp
            target = target
            filt = filt
            break
        else: 
            continue

 #   img_modl_PSNR = model_MODL(inp)
    img_modl_SSIM = model_SSIM(inp)
    js = [0,1,4,8]

    fig, axs = plt.subplots(2,len(js)*3-2, figsize = (12,3))
    
    x0 = img_modl_SSIM['dc0'].detach().cpu().numpy()[0,0,...]

    img = img_modl_SSIM
    
    axs[0,0].imshow(img['dc0'].detach().cpu().numpy()[0,0,...], cmap = 'gray')
    axs[1,0].imshow(img['dc0'].detach().cpu().numpy()[0,0,30:60, 30:60], cmap = 'gray') 
    
    rect = patches.Rectangle((30, 30), 30, 30, linewidth=1, edgecolor='r', facecolor='none')
    axs[0, 0].add_patch(rect)
    axs[1, 0].set_xlabel('DC0')

    for spine in axs[1, 0].spines.values():
        spine.set_edgecolor('red')

    for i,j in zip(range(1,len(js)*3, 3), js[1:]):
        
        j = str(j)

        axs[0,i].imshow(img['dw'+j].detach().cpu().numpy()[0,0,...], cmap = 'gray')
        axs[1, i].imshow(img['dw'+j].detach().cpu().numpy()[0,0,30:60, 30:60], cmap = 'gray') 
        rect = patches.Rectangle((30, 30), 30, 30, linewidth=1, edgecolor='r', facecolor='none')
        axs[0, i].add_patch(rect)
        axs[1, i].tick_params(color = 'r') 
        for spine in axs[1, i].spines.values():
            spine.set_edgecolor('red')
        axs[1, i].set_xlabel(r'$\mathcal{D}_w$'+j)
            
        nw = img['dc'+str(int(j)-1)].detach().cpu().numpy()[0,0,...]-img['dw'+j].detach().cpu().numpy()[0,0,...]
        axs[0,i+1].imshow(nw, cmap = 'gray')
        axs[1, i+1].imshow(nw[30:60, 30:60], cmap = 'gray') 
        rect = patches.Rectangle((30, 30), 30, 30, linewidth=1, edgecolor='r', facecolor='none')
        axs[0, i+1].add_patch(rect)
        axs[1, i+1].tick_params(color = 'r') 
        for spine in axs[1, i+1].spines.values():
            spine.set_edgecolor('red')
        axs[1,i+1].set_xlabel(r'$\mathcal{N}_w$'+j)
        
        axs[0,i+2].imshow(img['dc'+j].detach().cpu().numpy()[0,0,...], cmap = 'gray')
        axs[1, i+2].imshow(img['dc'+j].detach().cpu().numpy()[0,0,30:60, 30:60], cmap = 'gray') 
        rect = patches.Rectangle((30, 30), 30, 30, linewidth=1, edgecolor='r', facecolor='none')
        axs[0, i+2].add_patch(rect)
        for spine in axs[1, i+2].spines.values():
            spine.set_edgecolor('red')
        axs[1, i+2].set_xlabel('DC'+j) 
    
    axs = axs.flatten()
    for ax in axs:
        ax.set_yticklabels([])
        ax.set_xticklabels([])

    fig.savefig(results_folder+'Denoising_Efect.pdf', bbox_inches = 'tight')

if __name__ == '__main__':

    denoise_stages()

