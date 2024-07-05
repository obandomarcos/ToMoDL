# %%
import os
import os, sys
from config import * 

sys.path.append(where_am_i())

import wandb
import cv2
from skimage.metrics import structural_similarity as ssim
from models.models_system import MoDLReconstructor, UNetReconstructor
from models import alternating as altmodels
from pathlib import Path
import numpy as np
from config import *
from utilities import dataloading_utilities as dlutils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from torchvision import transforms as T
from pytorch_msssim import SSIM
# from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure as MSSSIM
from skimage.transform import radon, iradon
import matplotlib.patches as patches
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
acceleration_factor = 20

# %%
# ResNet dictionary parameters
resnet_options_dict = {'number_layers': 8,
                    'kernel_size':3,
                    'features':64,
                    'in_channels':1,
                    'out_channels':1,
                    'stride':1, 
                    'use_batch_norm': True,
                    'init_method': 'xavier'}

# Model parameters
modl_dict = {'use_torch_radon': False,
            'metric': 'psnr',
            'K_iterations' : 8,
            'number_projections_total' : 720,
            'acceleration_factor': acceleration_factor,
            'image_size': 100,
            'lambda': 0.025,
            'use_shared_weights': True,
            'denoiser_method': 'resnet',
            'resnet_options': resnet_options_dict,
            'in_channels': 1,
            'out_channels': 1}

admm_dictionary = {'number_projections': modl_dict['number_projections_total'],
                    'alpha': 0.005, 
                    'delta': 2, 
                    'max_iter': 10, 
                    'tol': 10e-7, 
                    'use_invert': 0,
                    'use_warm_init' : 1,
                    'verbose': True}

twist_dictionary = {'number_projections': modl_dict['number_projections_total'], 
                        'lambda': 1e-4, 
                        'tolerance':1e-4,
                        'stop_criterion':1, 
                        'verbose':0,
                        'initialization':0,
                        'max_iter':10000, 
                        'gpu':0,
                        'tau': 0.02}

# Training parameters
loss_dict = {'loss_name': 'psnr',
            'psnr_loss': torch.nn.MSELoss(reduction = 'mean'),
            'ssim_loss': SSIM(data_range=1, size_average=True, channel=1),
            'msssim_loss': MSSSIM(kernel_size = 1),
            'l1_loss' : torch.nn.L1Loss(reduction = 'mean')}

# Optimizer parameters
optimizer_dict = {'optimizer_name': 'Adam',
                'lr': 1e-4}

unet_dict = {'n_channels': 1,
            'n_classes':1,
            'bilinear': True,
            'batch_norm': True,
            'batch_norm_inconv':True,
            'residual': False,
            'up_conv': False}

# System parameters
model_system_dict = {'acc_factor_data': 1,
                'use_normalize': True,
                'optimizer_dict': optimizer_dict,
                'kw_dictionary_modl': modl_dict,
                'kw_dictionary_unet': unet_dict,
                'loss_dict': loss_dict, 
                'method':'modl',                       
                'track_train': True,
                'track_val': True,
                'track_test': True,
                'max_epochs': 25, 
                'save_model':True,
                'load_model': False,
                'load_path': '',
                'save_path': 'MoDL_K_fold_{}',
                'track_alternating_admm':False,
                'tv_iters': 3,
                'title': 'HyperParams_Search',
                'metrics_folder': where_am_i('metrics'),
                'models_folder': where_am_i('models'),
                'track_alternating_admm': False,         
                'track_alternating_twist': False,
                'track_unet': False}
# %%
hR = lambda x: radon(x, angles, circle = False)
hRT = lambda sino: iradon(sino, angles, circle = False)
Psi = lambda x,th: altmodels.TVdenoise(x,
                                2/th,
                                model_system_dict['tv_iters'])

Phi = lambda x: altmodels.TVnorm(x)

angles = np.linspace(0, 2*180, modl_dict['number_projections_total'] , endpoint = False)

kwarg = {'PSI': Psi, 'PHI': Phi, 'LAMBDA':twist_dictionary['lambda'], 'TOLERANCEA': twist_dictionary['tolerance'], 'STOPCRITERION': twist_dictionary['stop_criterion'], 'VERBOSE': twist_dictionary['verbose'], 'INITIALIZATION': twist_dictionary['initialization'], 'MAXITERA':twist_dictionary['max_iter'], 'GPU' : twist_dictionary['gpu']}

TwIST = lambda y, y_true: altmodels.TwIST(y = hR(y), 
                                            A = hR, 
                                            AT = hRT, 
                                            tau = twist_dictionary['tau'], 
                                            kwarg = kwarg,true_img = y_true)


# %% 20% percent trained on 20
run = wandb.init()
artifact_tomodl = run.use_artifact('omarcos/PSNR - Training Samples/model-uqb3ptp0:v0', type='model')
artifact_tomodl_dir = artifact_tomodl.download()
model_tomodl = MoDLReconstructor.load_from_checkpoint(Path(artifact_tomodl_dir) / "model.ckpt", kw_dictionary_model_system = model_system_dict)

model_system_dict['method'] = 'unet'
# 20% percent trained on 20
# run = wandb.init()
artifact_unet = run.use_artifact('omarcos/Unet - Training Samples/model-a3be8n7l:v0', type='model')
artifact_unet_dir = artifact_unet.download()
model_unet = UNetReconstructor.load_from_checkpoint(Path(artifact_unet_dir) / "model.ckpt", kw_dictionary_model_system = model_system_dict)
# %%
dataset_list_generic = ['140315_3dpf_head_{}', '140114_5dpf_head_{}', '140519_5dpf_head_{}', '140117_3dpf_body_{}', '140114_5dpf_upper tail_{}', 
'140315_1dpf_head_{}', '140114_5dpf_lower tail_{}', '140714_5dpf_head_{}', '140117_3dpf_head_{}', '140117_3dpf_lower tail_{}', '140117_3dpf_upper tail_{}', '140114_5dpf_body_{}']
dataset_list = [where_am_i('datasets')+'x{}/'.format(acceleration_factor)+dataset.format(acceleration_factor) for dataset in dataset_list_generic]

dataset_dict = {'root_folder' : dataset_list[1], 
                'acceleration_factor' : acceleration_factor,
                'transform' : None}
# %%
# %%

def psnr(img1, img2, range_max_min = [0,1]):
    mse = ((img1-img2)**2).mean()
    
    return round(10*np.log10((range_max_min[1]-range_max_min[0])**2/mse), 2)

def normalize_image_std(image):
        
    image_norm = 255.0*(image - image.mean())/(image.std())
    # image_norm = ((image - image.min())/(image.max()-image.min()))
    return image_norm   

def normalize_image(image):
        
    # image = cv2.equalizeHist((255.0*(image - image.min())/(image.max()-image.min())).astype(np.uint8))
    image_norm = (image - image.min())/(image.max()-image.min())
    return image_norm  
# %%
test_dataset = dlutils.ReconstructionDataset(**dataset_dict)    

test_dataloader = DataLoader(test_dataset, 
                            batch_size = 1,
                            shuffle = False,
                            num_workers = 0)
models = [model_tomodl, model_unet]
# %%

for i, batch in enumerate(test_dataloader):
    if i == 300:
        break
    us_unfil_im, us_fil_im, fs_fil_im = batch

im_tomodl= normalize_image(models[0](us_unfil_im.to(device))['dc'+str(model_tomodl.model.K)][0,0,...].detach().cpu().numpy())
im_unet= normalize_image(models[1](us_unfil_im.to(device))[0,0,...].detach().cpu().numpy())
us_unfil_im = normalize_image(us_unfil_im[0,0,...].detach().cpu().numpy())


im_twist = normalize_image(TwIST(us_fil_im.to(device).detach().cpu().numpy()[0, 0,...].T, fs_fil_im.to(device).detach().cpu().numpy()[0, 0,...].T)[0].T)

us_fil_im = normalize_image(us_fil_im[0,0,...].detach().cpu().numpy())
fs_fil_im = normalize_image(fs_fil_im[0,0,...].detach().cpu().numpy())


value = -0.135
im_unet = np.where((1.0 - im_unet) < value,1.0, im_unet+value)

value = 0.1
im_tomodl = np.where((1.0 - im_tomodl) < value,1.0, im_tomodl+value)

# %%
bx, tx = (50,90)
by, ty = (20,60)

images = [[im_tomodl, im_unet, im_twist, us_unfil_im, us_fil_im, fs_fil_im], [im_tomodl[ by:ty, bx:tx], im_unet[ by:ty, bx:tx], im_twist[by:ty, bx:tx], us_unfil_im[ by:ty, bx:tx], us_fil_im[by:ty, bx:tx], fs_fil_im[ by:ty, bx:tx,]]]


# %%
fig, axes = plt.subplots(2,6, figsize = (9,3))

for enum, (axs, (im_tomodl, im_unet, im_twist, us_unfil_im, us_fil_im, fs_fil_im)) in enumerate(zip(axes, images)):
    
    axs[0].imshow(us_unfil_im, cmap= 'magma', vmin = 0, vmax = 1)
    plt.setp(axs[0].spines.values(), visible=False)
    axs[0].tick_params(left=False, bottom = False, labelbottom=False, labelleft=False)

    axs[1].imshow(us_fil_im, cmap= 'magma', vmin = 0, vmax = 1)
    plt.setp(axs[1].spines.values(), visible=False)
    axs[1].tick_params(left=False, bottom = False, labelbottom=False, labelleft=False)
    
    axs[2].imshow(im_twist, cmap = 'magma', vmin = 0, vmax = 1)
    plt.setp(axs[2].spines.values(), visible=False)
    axs[2].tick_params(left=False, bottom = False, labelbottom=False, labelleft=False)
    
    axs[3].imshow(im_unet, cmap = 'magma', vmin = 0, vmax = 1)
    plt.setp(axs[3].spines.values(), visible=False)
    axs[3].tick_params(left=False, bottom = False, labelbottom=False, labelleft=False)
    
    axs[4].imshow(im_tomodl, cmap= 'magma', vmin = 0, vmax = 1)
    plt.setp(axs[4].spines.values(), visible=False)
    axs[4].tick_params(left=False, bottom = False, labelbottom=False, labelleft=False) 

    axs[5].imshow(fs_fil_im, cmap= 'magma', vmin = 0, vmax = 1)
    plt.setp(axs[5].spines.values(), visible=False)
    axs[5].tick_params(left=False, bottom = False, labelbottom=False, labelleft=False)
    
    if enum == 1:

        ssim_us = round(ssim(images[0][4], images[0][5]), 3)
        ssim_tomodl = round(ssim(images[0][0], images[0][5]), 3)
        ssim_unet = round(ssim(images[0][1], images[0][5]), 3)
        ssim_twist = round(ssim(images[0][2], images[0][5]), 3)

        us_unfil_im = normalize_image_std(images[0][3])
        us_fil_im = normalize_image_std(images[0][4])
        fs_fil_im = normalize_image_std(images[0][5])
        im_tomodl = normalize_image_std(images[0][0])
        im_unet = normalize_image_std(images[0][1])
        im_twist = normalize_image_std(images[0][2])

        axs[0].set_xlabel(r'$\mathbf{A}^\mathrm{H}\mathrm{b}$'+' 20x')
        axs[1].set_xlabel('FBP 20x'+f'\nPSNR {psnr(us_fil_im, fs_fil_im, [fs_fil_im.min(), fs_fil_im.max()])} dB'+f'\nSSIM: {ssim_us}')
        axs[2].set_xlabel(r'TwIST 20x'+f'\nPSNR {psnr(im_twist, fs_fil_im, [fs_fil_im.min(), fs_fil_im.max()])} dB'+f'\nSSIM: {ssim_twist}')
        axs[3].set_xlabel(r'U-Net 20x'+f'\nPSNR {psnr(im_unet, fs_fil_im,[im_unet.min(), im_unet.max()])} dB'+f'\nSSIM: {ssim_unet}')
        axs[4].set_xlabel(r'ToMoDL 20x'+f'\nPSNR {psnr(im_tomodl, fs_fil_im, [im_tomodl.min(), im_tomodl.max()])} dB'+f'\nSSIM: {ssim_tomodl}')
        axs[5].set_xlabel('FBP 1x\nGround truth')

    if enum == 0:

        fs_fil_im = normalize_image(fs_fil_im)
        im = axs[5].imshow(fs_fil_im, cmap= 'magma')
        for ax in axs:
            rect = patches.Rectangle((bx, by), tx-bx, ty-by, linewidth=1, edgecolor='orange',     facecolor='none')
            # Add the patch to the Axes
            ax.add_patch(rect)
    
    if enum == 1:
        
        for ax in axs:
            plt.setp(ax.spines.values(), visible=True)
            ax.spines['bottom'].set_color('orange')
            ax.spines['top'].set_color('orange') 
            ax.spines['right'].set_color('orange')
            ax.spines['left'].set_color('orange')


cax = fig.add_axes([axs[5].get_position().x1+0.01,axs[5].get_position().y0+0.04,0.02,2*axs[5].get_position().height])
im = plt.colorbar(im, cax = cax)

fig.savefig('/home/obanmarcos/Balseiro/DeepOPT/results/27-Invariance.pdf', bbox_inches = 'tight')
