"""
Plot different figures for Mendoza's poster
"""
import os
import os,time, sys
os.chdir('/home/marcos/DeepOPT/')
sys.path.append('Utilities/')
sys.path.append('OPTmodl/')
sys.path.append('Reconstruction/')

import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import DataLoading as DL
from Folders_cluster import *
import ModelUtilities as modutils
import Reconstruction as RecTV
import torch
from torch_radon import Radon, RadonFanbeam
from skimage.transform import radon, iradon
import phantominator as ph
import torchvision
import model_torch as modl
import pickle
from tqdm import tqdm
import cv2
from skimage.metrics import structural_similarity as ssim

# Using CPU or GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
folder_paths = [ f140315_3dpf] # Folders to be used

umbral_reg = 50

#%% Datasets 
# Training with more than one dataset
proj_num = 72

train_factor = 0.7
val_factor = 0.2
test_factor = 0.1
total_size = 3000

lam = 0.05
nLayer= 8
K = 10
epochs = 30
max_angle = 640

batch_size = 5
img_size = 100
augment_factor = 10
train_infos = {}
test_loss_dict = {}
tensor_path = datasets_folder+'Proj_{}_augmentFactor_{}_totalSize_{}_'.format(proj_num, augment_factor, total_size)

datasets_loaded = modutils.formRegDatasets(folder_paths, umbral_reg, img_resize = img_size)
datasets = []

dataloaders = modutils.formDataloaders(datasets, proj_num, total_size, train_factor, val_factor, test_factor, batch_size, img_size, tensor_path, augment_factor, load_tensor = True, save_tensor = False)

#2 - Comparison with Iterative methods
tv_iters = 3
full_sino = datasets_loaded[0]
undersampled_sino = np.copy(datasets_loaded[0])

# Zero pad
det_count = int((img_size+0.5)*np.sqrt(2))
num_beams = 72
zeros_idx = np.linspace(0, full_sino.shape[0], num_beams, endpoint = False).astype(int)
zeros_mask = np.full(full_sino.shape[0], True, dtype = bool)
zeros_mask[zeros_idx] = False
undersampled_sino[zeros_mask, :, :] = 0

n_angles = full_sino.shape[0]
theta = np.linspace(0, 360.0, n_angles, endpoint = False)
angles = np.linspace(0, 2*np.pi, n_angles, endpoint = False)

Psi = lambda x,th:  RecTV.TVdenoise(x,2/th,tv_iters)
#  set the penalty function, to compute the objective
Phi = lambda x: RecTV.TVnorm(x)
hR = lambda x: radon(x, theta, circle = False)
hRT = lambda sino: iradon(sino, theta, circle = False)
rad_GPU = Radon(img_size, angles, clip_to_circle = False, det_count = det_count)
train_name = 'Optimization_Layers_Test29'

model_name = model_folder+train_name+'K_{}_lam_{}_nlay_{}_proj_{}'.format(K, lam, nLayer, proj_num)
model = modl.OPTmodl(nLayer, K, max_angle, proj_num, img_size, None, lam, True, results_folder)
modutils.load_net(model_name, model, device)

tolA = 1e-4
kwargs = {'Lambda':1e-4,
         'AT':hRT,
         'Psi':Psi,
         'Phi':Phi, 
         'GPU' : 0,
         'Monotone':1,
         'MaxiterA':10000,
         'Initialization':0,
         'StopCriterion':1,
       	 'ToleranceA':tolA,
         'Verbose': 1}

times_ADMM = []
times_MODL = []
times_FBP_GPU = []
times_FBP_CPU = []

psnr_ADMM = []
psnr_MODL = []
psnr_FBP = []

ssim_ADMM = []
ssim_MODL = []
ssim_FBP = []

for i,(sino_under, sino_full) in enumerate(zip(np.rollaxis(undersampled_sino, 2), np.rollaxis(full_sino, 2))): 

    if (i<100) or (i>101):
        continue

    true_img = rad_GPU.backward(rad_GPU.filter_sinogram(torch.Tensor(sino_full).to(device))).detach().cpu().numpy()
    true_img = (true_img-true_img.min())/(true_img.max()-true_img.min())

    # GPU model - A^Tb -> DEEPOPT
    model_input = torch.FloatTensor(sino_under).to(device)
    model_input = (model_input - model_input.min())/(model_input.max()-model_input.min())
    model_input_img = rad_GPU.backward(model_input)*np.pi/n_angles
    model_input_img = (model_input_img - model_input_img.min())/(model_input_img.max()-model_input_img.min())
    model_input_img = torch.unsqueeze(torch.unsqueeze(model_input_img, 0), 0)    

    # Benchmarking
    time_MODL = time.time()
    x_MODL = model(model_input_img)['dc10']
    #x_MODL = (x_MODL-x_MODL.min())/(x_MODL.max()-x_MODL.min())
    time_MODL = time.time()-time_MODL
    
    time_ADMM = time.time()    
    sino_under_ADMM = np.copy((sino_under - sino_under.min())/(sino_under.max()-sino_under.min()))
    x_ADMM, _, _, _ = RecTV.TwIST(sino_under.T, hR, hRT, 0.001, kwargs,true_img = true_img)
    x_ADMM = (x_ADMM-x_ADMM.min())/(x_ADMM.max()-x_ADMM.min())
    time_ADMM = time.time() - time_ADMM
    
    time_FBP_GPU = time.time()
    x_FBP_GPU = rad_GPU.backward(rad_GPU.filter_sinogram(model_input))
    time_FBP_GPU = time.time()-time_FBP_GPU 

    time_FBP_CPU = time.time()
    x_FBP_CPU = hRT(sino_under.T)
    x_FBP_CPU = (x_FBP_CPU - x_FBP_CPU.min())/(x_FBP_CPU.max()-x_FBP_CPU.min())
    time_FBP_CPU = time.time()-time_FBP_CPU 

    times_MODL.append(time_MODL)
    times_ADMM.append(time_ADMM)
    times_FBP_GPU.append(time_FBP_GPU)
    times_FBP_CPU.append(time_FBP_CPU)
    
    mse_MODL = ((x_MODL.detach().cpu().numpy()-true_img)**2).sum()/true_img.size
    mse_ADMM = ((x_ADMM-true_img)**2).sum()/true_img.size
    mse_FBP = ((x_FBP_CPU-true_img)**2).sum()/true_img.size

    psnr_MODL.append(10*np.log10(x_MODL.detach().cpu().numpy().max()**2/mse_MODL))
    psnr_ADMM.append(10*np.log10(x_ADMM.max()**2/mse_ADMM))
    psnr_FBP.append(10*np.log10(x_FBP_CPU.max()**2/mse_FBP))

    print(x_MODL.detach().cpu().numpy().shape, true_img.shape)
    #ssim_ADMM.append(ssim(x_ADMM, true_img))
    #ssim_MODL.append(ssim(x_MODL.detach().cpu().numpy(), true_img))
    #ssim_FBP.append(ssim(x_FBP_CPU, true_img))


print('tiempos MODL', times_MODL)
print('tiempos ADMM', times_ADMM)
print('tiempos FBP CPU', times_FBP_CPU)
print('tiempos FBP GPU', times_FBP_GPU)

print('---------------\n')
print('PSNR MODL', psnr_MODL)
print('PSNR ADMM', psnr_ADMM)
print('PSNR FBP CPU', psnr_FBP)

print('---------------\n')
print('SSIM MODL', ssim_MODL)
print('SSIM MODL', ssim_ADMM)
print('SSIM MODL', ssim_FBP)

fig, ax = plt.subplots(1,3)
ax[0].imshow(x_MODL.detach().cpu().numpy()[0,0,...])
ax[0].set_title('MODL')
ax[1].imshow(x_ADMM)
ax[1].set_title('ADMM')
ax[2].imshow(x_FBP_CPU)
ax[2].set_title('FBP')

fig.savefig(results_folder+'Test32_Comparison.pdf', bbox_inches = 'tight')

inp = next(iter(dataloaders['test']['x']))
fbp = next(iter(dataloaders['test']['filtX']))

fig, ax = plt.subplots(1,2)
im1 = ax[0].imshow(inp[0,0,...].detach().cpu().numpy())
plt.colorbar(im1)
im2 = ax[1].imshow(model_input[0,0,...].detach().cpu().numpy())
plt.colorbar(im2)
fig.savefig(results_folder+'Test33_MODL.pdf')
