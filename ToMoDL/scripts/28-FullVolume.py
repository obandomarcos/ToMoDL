#%%
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
from torchmetrics import MultiScaleStructuralSimilarityIndexMeasure as MSSSIM
from skimage.transform import radon, iradon
import matplotlib.patches as patches
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
acceleration_factor = 20


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
                    'lambda': 1e-6, 
                    'tolerance':1e-6,
                    'stop_criterion':1, 
                    'verbose':1,
                    'initialization':0,
                    'max_iter':100, 
                    'gpu':0,
                    'tau': 0.01}

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

#%% 20% percent trained on 20
run = wandb.init()
artifact_tomodl = run.use_artifact('omarcos/PSNR - Training Samples/model-uqb3ptp0:v0', type='model')
artifact_tomodl_dir = artifact_tomodl.download()
model_tomodl = MoDLReconstructor.load_from_checkpoint(Path(artifact_tomodl_dir) / "model.ckpt", kw_dictionary_model_system = model_system_dict)

model_system_dict['method'] = 'unet'
# 20% percent trained on 20
# run = wandb.init()
# artifact_unet = run.use_artifact('omarcos/deepopt/model-1pk6th7h:v0', type='model')
# artifact_unet_dir = artifact_unet.download()
# model_unet = UNetReconstructor.load_from_checkpoint(Path('/home/obanmarcos/Balseiro/DeepOPT/models/glowing-thunder-8/dbh5di08/checkpoints/') / "epoch=24-step=10000.ckpt", kw_dictionary_model_system = model_system_dict)

dataset_list = ['/home/obanmarcos/Balseiro/DeepOPT/datasets/full_fish/x20/140415_5dpf_4X_head_20']
# dataset_list = [where_am_i('datasets')+'x{}/'.format(acceleration_factor)+dataset.format(acceleration_factor) for dataset in dataset_list_generic]

dataset_dict = {'root_folder' : dataset_list[0], 
                'acceleration_factor' : acceleration_factor,
                'transform' : None}
#%%
test_dataset = dlutils.ReconstructionDataset(**dataset_dict)    

test_dataloader = DataLoader(test_dataset, 
                            batch_size = 1,
                            shuffle = False,
                            num_workers = 0)

vol_tomodl = []
vol_fbp = []

for enum, (us_unfil_im, us_fil_im, fs_fil_im) in enumerate(test_dataloader):
    
    im_tomodl= model_tomodl(us_unfil_im.to(device))['dc'+str(model_tomodl.model.K)][0,0,...].detach().cpu().numpy()
    im_fbp = us_fil_im[0,0,...].detach().cpu().numpy()
    im_truth = fs_fil_im[0,0,...].detach().cpu().numpy()

    scale_percent = 400 # percent of original size
    width = int(im_tomodl.shape[1] * scale_percent / 100)
    height = int(im_tomodl.shape[0] * scale_percent / 100)
    dim = (width, height)

    im_tomodl = cv2.resize(im_tomodl, dim, interpolation = cv2.INTER_AREA)
    im_fbp = cv2.resize(im_fbp, dim, interpolation = cv2.INTER_AREA)
    im_truth = cv2.resize(im_truth, dim, interpolation = cv2.INTER_AREA)
    
    print(cv2.imwrite(f'/home/obanmarcos/Balseiro/DeepOPT/Volumes/X20_Tomodl/{enum}.jpg', 255.0*im_tomodl))
    print(cv2.imwrite(f'/home/obanmarcos/Balseiro/DeepOPT/Volumes/X20_fbp/a_{enum}.jpg', 255.0*im_fbp))
    print(cv2.imwrite(f'/home/obanmarcos/Balseiro/DeepOPT/Volumes/X20_fbp_GT/a_truth_{enum}.jpg', 255.0*im_truth))
    


