'''
Load K-folding models from groups and evaluate performance

author : obanmarcos
'''
import os
import os, sys
from config import * 

sys.path.append(where_am_i())

import pytorch_lightning as pl
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utilities import dataloading_utilities as dlutils
from utilities.folders import *

from training import train_utilities as trutils

from models.models_system import MoDLReconstructor, UNetReconstructor, TwISTReconstructor, FBPReconstructor
import torch

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger


from torch.utils.data import DataLoader
from torchvision import transforms as T
from pytorch_msssim import SSIM
# from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure as MSSSIM
import wandb
from pathlib import Path
import cv2

group_name = ''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

use_default_model_dict = True
use_default_dataloader_dict = True
use_default_trainer_dict = True

def load_dicts(acceleration_factor, use_default_model_dict = True, use_default_dataloader_dict = True ,use_default_trainer_dict = True):

    if use_default_model_dict == True:
       
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

        twist_dictionary = {'number_projections': modl_dict['number_projections_total'], 
                'lambda': 1e-2, 
                'tolerance':1e-4,
                'stop_criterion':1, 
                'verbose':0,
                'initialization':0,
                'max_iter':10, 
                'gpu':0,
                'tau': 0.01}
        
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
                        'tv_iters': 20,
                        'title': 'HyperParams_Search',
                        'metrics_folder': where_am_i('metrics'),
                        'models_folder': where_am_i('models'),
                        'track_alternating_admm': False,         
                        'track_alternating_twist': True,
                        'track_unet': False,
                        'twist_dictionary':twist_dictionary}
        
    # PL Trainer and W&B logger dictionaries
    if use_default_trainer_dict == True:
                
        logger_dict = {'project':'deepopt',
                        'entity': 'omarcos', 
                        'log_model': True}

        lightning_trainer_dict = {'max_epochs': 25,
                                'log_every_n_steps': 10,
                                'check_val_every_n_epoch': 1,
                                'gradient_clip_val' : 1,
                                'accelerator' : 'gpu', 
                                'devices' : 1,
                                'fast_dev_run' : False,
                                'default_root_dir': where_am_i('models')}

        profiler = None
        # profiler = SimpleProfiler(dirpath = './logs/', filename = 'Test_training_profile_pytorch')
        # profiler = PyTorchProfiler(dirpath = './logs/', filename = 'Test_training_profile_pytorch')

        trainer_dict = {'lightning_trainer_dict': lightning_trainer_dict,
                        'use_k_folding': True, 
                        'track_checkpoints': True,
                        'epoch_number_checkpoint': 10,
                        'use_swa' : False,
                        'use_accumulate_batches': False,
                        'k_fold_number_datasets': 3,
                        'use_logger' : True,
                        'logger_dict': logger_dict,
                        'track_default_checkpoints'  : True,
                        'use_auto_lr_find': False,
                        'batch_accumulate_number': 3,
                        'use_mixed_precision': False,
                        'batch_accumulation_start_epoch': 0, 
                        'profiler': profiler, 
                        'restore_fold': False,
                        'resume': False}

    # Dataloader dictionary
    if use_default_dataloader_dict == True:
        
        # data_transform = T.Compose([T.ToTensor()])
        data_transform = None                                    
        
        dataloader_dict = {'datasets_folder': where_am_i('datasets'),
                        'number_volumes' : 0,
                        'experiment_name': 'Bassi',
                        'img_resize': 100,
                        'load_shifts': True,
                        'save_shifts':False,
                        'number_projections_total': 720,
                        'number_projections_undersampled': 72,
                        'acceleration_factor':acceleration_factor,
                        'train_factor' : 0.8, 
                        'val_factor' : 0.2,
                        'test_factor' : 0.2, 
                        'batch_size' : 8, 
                        'sampling_method' : 'equispaced-linear',
                        'shuffle_data' : True,
                        'data_transform' : data_transform,
                        'num_workers':0,
                        'use_number_samples':False,
                        'number_samples_factor':1.0,
                        'use_subset_by_part': False}
        
        return trainer_dict, dataloader_dict, model_system_dict

def load_runs(user_project_name, models_methods, acceleration_factors):

    filter_dict = {}
    api = wandb.Api()
    runs = api.runs(path = user_project_name, filters=filter_dict)
    
    # Form dict runs for analysis
    dict_runs = {k:{'x'+str(k):{} for k in acceleration_factors} for k in models_methods}
    
    # Read runs from W&B
    for run in runs:

        if not run.name[1:3].replace(' ', '').isdigit():

            continue

        if int(run.name[1:3].replace(' ', '')) in acceleration_factors:

            if 'psnr' in run.name:
                
                for method in models_methods:

                    if 'kw_dictionary_modl/metric' in run.config and method == 'tomodl':
                    
                        dict_runs[method][run.name[:3].replace(' ', '')][run.name[-3:]] = run 

                    if 'kw_dictionary_unet/batch_norm' in run.config and method == 'unet':
                    
                        dict_runs[method][run.name[:3].replace(' ', '')][run.name[-3:]] = run 

    return dict_runs
def normalize_image(image):

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    image = (image-image.min())/(image.max()-image.min())
    return clahe.apply((255.0*image).astype(np.uint8))

if __name__ == '__main__':
    
    metric = 'psnr'
    
    dataset_list_generic = ['140315_3dpf_head_{}', '140114_5dpf_head_{}', '140519_5dpf_head_{}', '140117_3dpf_body_{}', '140114_5dpf_upper tail_{}', 
    '140315_1dpf_head_{}', '140114_5dpf_lower tail_{}', '140714_5dpf_head_{}', '140117_3dpf_head_{}', '140117_3dpf_lower tail_{}', '140117_3dpf_upper tail_{}', '140114_5dpf_body_{}']
    
    models_methods = ['tomodl', 'unet']
    min_acceleration, max_acceleration, step_acceleration = (12, 30, 4)
    acceleration_factors = np.arange(min_acceleration, max_acceleration, step_acceleration)
    user_project_name = 'omarcos/deepopt'
    model_types = ['fbp', 'twist', 'unet', 'tomodl', 'fbp_ground_truth', 'tomodl_denoiser']


    fig, axs = plt.subplots(len(model_types), len(acceleration_factors), figsize = (12,9))
    
    for acceleration_factor_idx, acceleration_factor in enumerate(acceleration_factors):
        
        # Form trainer
        trainer_dict, dataloader_dict, model_system_dict = load_dicts(acceleration_factor)
        dataset_list = [where_am_i('datasets')+'x{}/'.format(acceleration_factor)+dataset.format(acceleration_factor) for dataset in dataset_list_generic]

        trainer_system = trutils.TrainerSystem(trainer_dict, dataloader_dict, model_system_dict)

        # Load image    
        dataset_dict = {'root_folder' : dataset_list[1], 
            'acceleration_factor' : acceleration_factor,
            'transform' : None}
        
        test_dataset = dlutils.ReconstructionDataset(**dataset_dict)  

        test_dataloader = DataLoader(test_dataset, 
                                    batch_size = 1,
                                    shuffle = False,
                                    num_workers = 8)
        N = 100
        
        for _ in range(N):
            us_unfil_im, us_fil_im, fs_fil_im = next(iter(test_dataloader))
        
        
        for model_type_idx, model_type in enumerate(model_types):
            # Get Model
        
            if model_type == 'tomodl':
                # TODO call model
                artifact_tomodl = run.use_artifact('omarcos/PSNR - Training Samples/model-uqb3ptp0:v0', type='model')
                artifact_tomodl_dir = artifact_tomodl.download()
                model = MoDLReconstructor.load_from_checkpoint(Path(artifact_tomodl_dir) / "model.ckpt", kw_dictionary_model_system = model_system_dict) 
            
            if model_type == 'unet':
                
                run = wandb.init()
                artifact = run.use_artifact('omarcos/deepopt/model-yuk1krcc:v0', type = 'model')
                artifact_dir = artifact.download()
                model = UNetReconstructor.load_from_checkpoint(Path(artifact_dir) / "model.ckpt", kw_dictionary_model_system = model_system_dict) 
            
            if model_type == 'twist':

                model = TwISTReconstructor(model_system_dict)

            if model_type == 'fbp':
                model = FBPReconstructor(model_system_dict)
                
            if model_type == 'tomodl':
                axs[model_type_idx, acceleration_factor_idx].imshow(normalize_image(model.forward(us_unfil_im.to(device))['dc'+str(model.model.K)][0,0,...].detach().cpu().numpy()) , cmap = 'magma')
            elif model_type == 'unet':
                axs[model_type_idx, acceleration_factor_idx].imshow(normalize_image(model.forward(us_fil_im.to(device))[0,0,...].detach().cpu().numpy()), cmap = 'magma')
            elif model_type == 'twist':
                axs[model_type_idx, acceleration_factor_idx].imshow(normalize_image(model.forward(us_fil_im, fs_fil_im)[0]), cmap = 'magma')
            elif model_type == 'fbp':
                axs[model_type_idx, acceleration_factor_idx].imshow(normalize_image(us_fil_im[0,0,...].detach().cpu().numpy()), cmap = 'magma')
            elif model_type == 'tomodl_denoiser':
                axs[model_type_idx, acceleration_factor_idx].imshow(normalize_image(model.forward(us_unfil_im.to(device))['dc'+str(model.model.K-1)][0,0,...].detach().cpu().numpy()-model.forward(us_unfil_im.to(device))['dw'+str(model.model.K)][0,0,...].detach().cpu().numpy()), cmap = 'magma')
            elif model_type == 'fbp_ground_truth':
                axs[model_type_idx, acceleration_factor_idx].imshow(normalize_image(fs_fil_im[0,0,...].detach().cpu().numpy()), cmap = 'magma')

            plt.setp(axs[model_type_idx, acceleration_factor_idx].spines.values(), visible=False)
            axs[model_type_idx, acceleration_factor_idx].tick_params(left=False, bottom = False, labelbottom=False, labelleft=False)
        

            
    fig.savefig('/home/obanmarcos/Balseiro/DeepOPT/results/29-AccelerationFactorImages_denoiser.pdf', bbox_inches = 'tight')
