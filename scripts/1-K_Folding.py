'''
K-Folding script
author: obanmarcos
'''
import os
import os, sys

sys.path.append('/home/obanmarcos/Balseiro/DeepOPT/')

import pytorch_lightning as pl
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utilities import dataloading_utilities as dlutils
from utilities.folders import *

from training import train_utilities as trutils

from models.models_system import MoDLReconstructor
import torch

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from torchvision import transforms as T
from pytorch_msssim import SSIM

# Options for folding menu
use_default_model_dict = True
use_default_dataloader_dict = True
use_default_trainer_dict = True

def runs(testing_options):
    # Model dictionary
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
        modl_dict = {'use_torch_radon': True,
                    'number_layers': 8,
                    'K_iterations' : 8,
                    'number_projections_total' : 720,
                    'number_projections_undersampled' : 72, 
                    'image_size': 100,
                    'lambda': 0.05,
                    'use_shared_weights': True,
                    'denoiser_method': 'resnet',
                    'resnet_options': resnet_options_dict,
                    'in_channels': 1,
                    'out_channels': 1}

        # Training parameters
        loss_dict = {'loss_name': 'psnr',
                    'psnr_loss': torch.nn.MSELoss(reduction = 'mean'),
                    'ssim_loss': SSIM(data_range = 1, size_average= True, channel = 1)}

        # Optimizer parameters
        optimizer_dict = {'optimizer_name': 'Adam',
                        'lr': 1e-4}

        # System parameters
        model_system_dict = {'optimizer_dict': optimizer_dict,
                            'kw_dictionary_modl': modl_dict,
                            'loss_dict': loss_dict,                        
                            'track_train': True,
                            'track_val': True,
                            'track_test': True}
    
    # PL Trainer and W&B logger dictionaries
    if use_default_trainer_dict == True:
        
        logger_dict = {'project':'deepopt',
                        'entity': 'omarcos', 
                        'log_model': True}

        lightning_trainer_dict = {'max_epochs': 40,
                                  'log_every_n_steps':1000,
                                  'check_val_every_n_epoch': 1,
                                  'gradient_clip_val' : 1.0,
                                  'accelerator' : 'gpu', 
                                  'devices' : 1,
                                  'default_root_dir': model_folder}

        trainer_dict = {'lightning_trainer_dict': lightning_trainer_dict,
                        'use_k_folding': True, 
                        'k_fold_number_datasets': 2,
                        'use_logger' : True,
                        'logger_dict': logger_dict,
                        'track_default_checkpoints' : True}

    # Dataloader dictionary
    if use_default_dataloader_dict == True:
        
        data_transform = T.Compose([T.ToTensor()])
                                    
        dataloader_dict = {'datasets_folder': datasets_folder,
                           'number_volumes' : 0,
                           'experiment_name': 'Bassi',
                           'img_resize': 100,
                           'load_shifts': True,
                           'save_shifts':False,
                           'number_projections_total': 720,
                           'number_projections_undersampled': 72,
                           'train_factor' : 0.8, 
                           'val_factor' : 0.2,
                           'test_factor' : 0.2, 
                           'batch_size' : 5, 
                           'sampling_method' : 'equispaced-linear',
                           'shuffle_data' : True,
                           'data_transform' : data_transform}
    
    # Create Custom trainer
    if 'train_ssim' in testing_options:
        
        model_system_dict['loss_dict']['loss_name'] = 'ssim'

        trainer = trutils.TrainerSystem(trainer_dict, dataloader_dict, model_system_dict)
        trainer.k_folding()
    
    if 'train_psnr' in testing_options:
        
        model_system_dict['loss_dict']['loss_name'] = 'psnr'

        trainer = trutils.TrainerSystem(trainer_dict, dataloader_dict, model_system_dict)
        trainer.k_folding()

if __name__ == '__main__':

    k_folding_options = []

    parser = argparse.ArgumentParser(description='Do K-folding with different networks')

    parser.add_argument('--train_psnr', help = 'Train w/PSNR loss with optimal hyperparameters', action="store_true")
    parser.add_argument('--train_ssim', help = 'Train w/SSIM loss with optimal hyperparameters', action="store_true")
    
    args = parser.parse_args()

    if args.train_psnr:

        print('Training MODL with PSNR loss...')
        k_folding_options.append('train_psnr')
    
    if args.train_ssim:
        
        print('Training MODL with SSIM loss...')
        k_folding_options.append('train_ssim')
    
    runs(k_folding_options)