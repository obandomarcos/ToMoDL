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

from models.models_system import MoDLReconstructor
import torch

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from torchvision import transforms as T
from pytorch_msssim import SSIM
# from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics import MultiScaleStructuralSimilarityIndexMeasure as MSSSIM
import wandb
from pathlib import Path

group_name = ''

use_default_model_dict = True
use_default_dataloader_dict = True
use_default_trainer_dict = True

acceleration_factor = 2

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
                'msssim_loss': MSSSIM(kernel_size = 1)}

    # Optimizer parameters
    optimizer_dict = {'optimizer_name': 'Adam+Tanh',
                    'lr': 1e-4}

    # System parameters
    model_system_dict = {'acc_factor_data': 1,
                    'use_normalize': True,
                    'optimizer_dict': optimizer_dict,
                    'kw_dictionary_modl': modl_dict,
                    'loss_dict': loss_dict, 
                    'method':'modl',                       
                    'track_train': True,
                    'track_val': True,
                    'track_test': True,
                    'max_epochs': 20, 
                    'save_model':True,
                    'load_path': '',
                    'save_path': 'MoDL_K_fold_{}',
                    'track_alternating_admm':False,
                    'tv_iters': 40,
                    'title': 'HyperParams_Search',
                    'metrics_folder': where_am_i('metrics'),
                    'models_folder': where_am_i('models'),
                    'track_alternating_admm': False,         
                    'track_alternating_twist': False,
                    'track_unet': False}

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
                    'k_fold_number_datasets': 2,
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
                        'num_workers':16,
                        'use_subset_by_part': False}

artifact_names_x22_psnr = ['model-14cjgco7:v0']

dataset_list_x22 = ['140315_3dpf_head_{}', '140114_5dpf_head_{}', '140519_5dpf_head_{}', '140117_3dpf_body_{}', '140114_5dpf_upper tail_{}', '140315_1dpf_head_{}', '140114_5dpf_lower tail_{}', '140714_5dpf_head_{}', '140117_3dpf_head_{}', '140117_3dpf_lower tail_{}', '140117_3dpf_upper tail_{}', '140114_5dpf_body_{}']

if __name__ == '__main__':
    
    artifact_names = artifact_names_x22_psnr
    testing_name_group = 'x{}_test2'.format(acceleration_factor)

    run_name = 'test_metrics_kfold_x{}'.format(acceleration_factor)
    metric = 'psnr'
    dataset_list = dataset_list_x22 
    dataset_list = [where_am_i('datasets')+'x{}/'.format(acceleration_factor)+dataset.format(acceleration_factor) for dataset in dataset_list]

    user_project_name = 'omarcos/deepopt/'

    trainer_system = trutils.TrainerSystem(trainer_dict, dataloader_dict,model_system_dict)
    trainer_system.set_datasets_list(dataset_list)

    for k_fold, artifact_name in enumerate(artifact_names):
        
        run = wandb.init(project = 'deepopt', reinit = True, group = testing_name_group, job_type = 'Dataset Evaluation + K-Folding', name = '{} fold'.format(k_fold))
        
        train_dataloader, val_dataloader, test_dataloader = trainer_system.generate_K_folding_dataloader()

        artifact = run.use_artifact(user_project_name+artifact_name, type='model')
        artifact_dir = artifact.download()
        
        model = MoDLReconstructor.load_from_checkpoint(Path(artifact_dir) / "model.ckpt", kw_dictionary_model_system = model_system_dict) 

        trainer = trainer_system.create_trainer()

        test_dict = trainer.test(model = model, dataloaders = test_dataloader, verbose = True)    
        
        print(test_dict)
        # # TO-DO: Agregar U-Net y métodos alternantes

        for (key, number) in trainer_system.kfold_monitor_dict[trainer_system.current_fold]['test']['fish_part'].items():

            for _ in range(number):
                wandb.log({'fish_part': key, '{}/metric'.format(key): test_dict})
        
        for (key, number) in trainer_system.kfold_monitor_dict[trainer_system.current_fold]['test']['fish_dpf'].items():

            for _ in range(number):
                wandb.log({'fish_dpf': key, '{}/metric'.format(key): test_dict})

        wandb.log({'acc_factor': acceleration_factor,'k_fold': k_fold, 'psnr_modl': test_dict['test/psnr'], 'monitor': trainer_system.kfold_monitor_dict[trainer_system.current_fold]})
        wandb.log({'acc_factor': acceleration_factor, 'k_fold': k_fold, 'psnr_fbp': test_dict['test/psnr_fbp'],'monitor': trainer_system.kfold_monitor_dict[trainer_system.current_fold]})
        wandb.log({'acc_factor': acceleration_factor,'k_fold': k_fold, 'ssim_modl': test_dict['test/ssim'], 'monitor': trainer_system.kfold_monitor_dict[trainer_system.current_fold]})
        wandb.log({'acc_factor': acceleration_factor, 'k_fold': k_fold, 'ssim_fbp': test_dict['test/ssim_fbp'], 'monitor': trainer_system.kfold_monitor_dict[trainer_system.current_fold]})
        
        trainer_system.current_fold += 1
