'''
Sweep of hyperparams
* Bajar el número de iteraciones y layers 
author: obanmarcos
'''
import os
import os, sys
from config import * 

sys.path.append(where_am_i())

import pytorch_lightning as pl
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from utilities import dataloading_utilities as dlutils
from utilities.folders import *

from training import train_utilities as trutils

from models.models_system import MoDLReconstructor
import torch
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from torchvision import transforms as T
from pytorch_msssim import SSIM
# from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure as MSSSIM

def load_metrics(metrics_name, plot_path):

    with open(metrics_name, 'rb') as f:

        metrics = pickle.load(f)
    
    fig, ax = plt.subplots(1,1, figsize = (8,6))

    ax.plot([m.detach().cpu().numpy() for m in metrics['train_metric']['train/ssim']])

    fig.savefig(where_am_i('results')+'/'+plot_path)

if __name__ == '__main__':
    
    wandb.init(config = {'acc_factor_data': 1, 'loss_name': 'ssim'})

    # Options for folding menu
    use_default_model_dict = True
    use_default_dataloader_dict = True
    use_default_trainer_dict = True

    acceleration_factor = 20

    metric = 'psnr'

    if use_default_model_dict == True:
        # ResNet dictionary parameters
        resnet_options_dict = {'number_layers': 5,
                            'kernel_size':3,
                            'features':64,
                            'in_channels':1,
                            'out_channels':1,
                            'stride':1, 
                            'use_batch_norm': False,
                            'init_method': 'xavier'}

        # Model parameters
        modl_dict = {'use_torch_radon': False,
                    'metric': metric,
                    'K_iterations' : 8,
                    'number_projections_total' : 720,
                    'number_projections_undersampled' : 720//acceleration_factor, 
                    'acceleration_factor': acceleration_factor,
                    'image_size': 100,
                    'lambda': 0.05,
                    'use_shared_weights': True,
                    'denoiser_method': 'resnet',
                    'resnet_options': resnet_options_dict,
                    'in_channels': 1,
                    'out_channels': 1}
        
        admm_dictionary = {'number_projections': modl_dict['number_projections_undersampled'],
                        'tv_iters': 5,
                        'alpha': 0.01, 
                        'delta': 0.5, 
                        'max_iter': 10, 
                        'tol': 10e-7, 
                        'use_invert': 0,
                        'use_warm_init' : 1,
                        'verbose':False}
        
        twist_dictionary = {'number_projections': modl_dict['number_projections_undersampled'], 
                            'lambda': 1e-4, 
                            'tolerance':1e-4,
                            'stop_criterion':1, 
                            'verbose':0,
                            'initialization':0,
                            'max_iter':10000, 
                            'gpu':0,
                            'tau': 0.02}
        # Training parameters
        loss_dict = {'loss_name': wandb.config.loss_name,
                    'psnr_loss': torch.nn.MSELoss(reduction = 'mean'),
                    'ssim_loss': SSIM(data_range=1, size_average=True, channel=1),
                    'msssim_loss': MSSSIM(kernel_size = 1)}

        # Optimizer parameters
        optimizer_dict = {'optimizer_name': 'Adam',
                        'lr': 1e-4}

        # System parameters
        model_system_dict = {'acc_factor_data': wandb.config.acc_factor_data,
                            'use_normalize': True,
                            'optimizer_dict': optimizer_dict,
                            'kw_dictionary_modl': modl_dict,
                            'loss_dict': loss_dict, 
                            'method':'modl',                       
                            'track_train': True,
                            'track_val': True,
                            'track_test': True,
                            'max_epochs':20, 
                            'track_alternating_admm':False,
                            'tv_iters': 40,
                            'title': 'HyperParams_Search',
                            'metrics_folder': where_am_i('metrics'),
                            'models_folder': where_am_i('models'),
                            'admm_dictionary': admm_dictionary,         
                            'track_alternating_twist': False,
                            'twist_dictionary': twist_dictionary}

    # PL Trainer and W&B logger dictionaries
    if use_default_trainer_dict == True:

        logger_dict = {'project':'deepopt',
                        'entity': 'omarcos', 
                        'log_model': True}

        lightning_trainer_dict = {'max_epochs': 20,
                                'log_every_n_steps': 10,
                                'check_val_every_n_epoch': 1,
                                'gradient_clip_val' : 0.5,
                                'accelerator' : 'gpu', 
                                'devices' : 1,
                                'fast_dev_run' : False,
                                'default_root_dir': model_folder}

        profiler = None

        trainer_dict = {'lightning_trainer_dict': lightning_trainer_dict,
                        'use_k_folding': False, 
                        'track_checkpoints': False,
                        'epoch_number_checkpoint': 10,
                        'use_swa' : False,
                        'use_accumulate_batches': False,
                        'k_fold_number_datasets': 3,
                        'use_logger' : True,
                        'resume':'allow',
                        'logger_dict': logger_dict,
                        'track_default_checkpoints'  : False,
                        'use_auto_lr_find': False,
                        'batch_accumulate_number': 3,
                        'use_mixed_precision': False,
                        'batch_accumulation_start_epoch': 0, 
                        'profiler': profiler,
                        'restore_fold': False,
                        'fold_number_restore': 2,
                        'acc_factor_restore': 22}

    # Dataloader dictionary
    if use_default_dataloader_dict == True:
        
        data_transform = None                                    
        
        dataloader_dict = {'datasets_folder': where_am_i('datasets'),
                            'number_volumes' : 0,
                            'experiment_name': 'Bassi',
                            'img_resize': 100,
                            'load_shifts': True,
                            'save_shifts':False,
                            'number_projections_total': 720,
                            'number_projections_undersampled': 720//acceleration_factor,
                            'acceleration_factor':acceleration_factor,
                            'train_factor' : 0.8, 
                            'val_factor' : 0.2,
                            'test_factor' : 0.2, 
                            'batch_size' : 10, 
                            'sampling_method' : 'equispaced-linear',
                            'shuffle_data' : True,
                            'data_transform' : data_transform,
                            'num_workers' : 0}

    model_system_dict['loss_dict']['loss_name'] = 'psnr'
    
    hyperparameter_defaults = {'trainer_kwdict': trainer_dict,
                            'dataloader_kwdict' : dataloader_dict, 
                            'model_system_kwdict': model_system_dict}
    
    trainer = trutils.TrainerSystem(**hyperparameter_defaults)
    
    trainer.train_model()
