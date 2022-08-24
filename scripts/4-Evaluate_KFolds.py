'''
Evaluate K-Fold models from artifacts
author: obanmarcos
'''
import os, sys
from config import *

sys.path.append(where_am_i())

import pytorch_lightning as pl
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torchsummary import summary
from training import train_utilities as trutils
from utilities import dataloading_utilities as dlutils
from utilities.folders import *



from models.models_system import MoDLReconstructor
import torch

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from torchvision import transforms as T
from pytorch_msssim import SSIM

import wandb

# Options for folding menu
use_default_model_dict = True
use_default_dataloader_dict = True
use_default_trainer_dict = True

def eval_models(testing_options):
        # Model dictionary
    if use_default_model_dict == True:
        # ResNet dictionary parameters
        resnet_options_dict = {'number_layers': 8,
                            'kernel_size':3,
                            'features':64,
                            'in_channels':1,
                            'out_channels':1,
                            'stride':1, 
                            'use_batch_norm': False,
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
        model_system_dict = {'max_epochs':40,
                            'optimizer_dict': optimizer_dict,
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

    if 'load_run' in testing_options:

        run = wandb.init()
        artifact = run.use_artifact('omarcos/deepopt/model-1ud4xx3w:v0', type='model')
        artifact_dir = artifact.download()
        
        checkpoint_path = artifact_dir.replace('./', '') + '/' + [each for each in os.listdir(artifact_dir) if each.endswith('.ckpt')][0]

        print('Artifact directory:\n')
        print(artifact_dir)

        print('Artifact:\n')
        print(artifact)

        test_model = MoDLReconstructor.load_from_checkpoint(checkpoint_path, kw_dictionary_model_system = model_system_dict)
        print(test_model)

    return

if __name__ == '__main__':

    testing_options = []

    parser = argparse.ArgumentParser(description='Load run')

    parser.add_argument('--load_run', help = 'Train w/PSNR loss with optimal hyperparameters', action="store_true")
    
    args = parser.parse_args()

    if args.load_run:
        
        print('Loading runs...')
        testing_options.append('load_run')
    
    eval_models(testing_options)