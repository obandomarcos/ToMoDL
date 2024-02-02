'''
K-Folding script
author: obanmarcos
'''
from mmap import ACCESS_READ
import os
import os, sys
from config import * 

sys.path.append(where_am_i())

import wandb
import pytorch_lightning as pl
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utilities import dataloading_utilities as dlutils
from utilities.folders import *
import pickle
from pathlib import Path
from training import train_utilities as trutils

from models.models_system import MoDLReconstructor
import torch

from torch.utils.data import DataLoader

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from torchvision import transforms as T
from pytorch_msssim import SSIM
# from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics import MultiScaleStructuralSimilarityIndexMeasure as MSSSIM

# Options for folding menu
use_default_model_dict = True
use_default_dataloader_dict = True
use_default_trainer_dict = True

acceleration_factor = 10

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
                'metric': 'psnr',
                'number_layers': 8,
                'K_iterations' : 8,
                'number_projections_total' : 720,
                'acceleration_factor': acceleration_factor,
                'image_size': 100,
                'lambda': 0.05,
                'use_shared_weights': True,
                'denoiser_method': 'resnet',
                'resnet_options': resnet_options_dict,
                'in_channels': 1,
                'out_channels': 1}
    
    admm_dictionary = {'number_projections': modl_dict['number_projections_total'],
                    'alpha': 0.005, 
                    'delta': 1.5, 
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
                        'tau': 0.01}
    
    unet_dict = {'n_channels': 1,
                     'n_classes':1,
                     'bilinear': True,
                     'batch_norm': False,
                     'batch_norm_inconv': False,
                     'residual': False,
                     'up_conv': False}
     
    # Training parameters
    loss_dict = {'loss_name': 'psnr',
                'psnr_loss': torch.nn.MSELoss(reduction = 'mean'),
                'ssim_loss': SSIM(data_range=1, size_average=True, channel=1),
                'msssim_loss': MSSSIM(kernel_size = 1)}

    # Optimizer parameters
    optimizer_dict = {'optimizer_name': 'Adam+Tanh',
                    'lr': 1e-4}

    # System parameters
    model_system_dict = {'acc_factor_data':1,
                        'use_normalize': False,
                        'optimizer_dict': optimizer_dict,
                        'kw_dictionary_modl': modl_dict,
                        'loss_dict': loss_dict, 
                        'method':'modl',                 
                        'track_train': True,
                        'track_val': True,
                        'track_test': True,
                        'load_path':'',
                        'save_path':'',
                        'max_epochs':40, 
                        'tv_iters': 5,
                        'metrics_folder': where_am_i('metrics'),
                        'models_folder': where_am_i('models'),
                        'track_alternating_admm': False,
                        'admm_dictionary': admm_dictionary,
                        'track_alternating_twist': False,
                        'twist_dictionary': twist_dictionary,
                        'track_unet': False,
                        'unet_dictionary':unet_dict}

# PL Trainer and W&B logger dictionaries
if use_default_trainer_dict == True:

    logger_dict = {'project':'deepopt',
                    'entity': 'omarcos', 
                    'log_model': True}

    lightning_trainer_dict = {'max_epochs': 40,
                                'log_every_n_steps': 10,
                                'check_val_every_n_epoch': 1,
                                'gradient_clip_val' : 0.5,
                                'accelerator' : 'gpu', 
                                'devices' : 1,
                                'fast_dev_run' : False,
                                'default_root_dir': where_am_i('metrics')}

    profiler = None
    # profiler = SimpleProfiler(dirpath = './logs/', filename = 'Test_training_profile_pytorch')
    # profiler = PyTorchProfiler(dirpath = './logs/', filename = 'Test_training_profile_pytorch')

    trainer_dict = {'lightning_trainer_dict': lightning_trainer_dict,
                    'use_k_folding': True, 
                    'track_checkpoints': False,
                    'epoch_number_checkpoint': 10,
                    'use_swa' : False,
                    'use_accumulate_batches': False,
                    'k_fold_number_datasets': 3,
                    'use_logger' : False,
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
    
    # data_transform = T.Compose([T.ToTensor()])
    data_transform = None                                    
    
    dataloader_dict = {'datasets_folder': where_am_i('datasets'),
                        'number_volumes' : 0,
                        'experiment_name': 'Bassi',
                        'img_resize': 100,
                        'load_shifts': True,
                        'save_shifts':False,
                        'number_projections_total': 720,
                        'acceleration_factor':acceleration_factor,
                        'train_factor' : 0.8, 
                        'val_factor' : 0.2,
                        'test_factor' : 0.2, 
                        'batch_size' : 8, 
                        'sampling_method' : 'equispaced-linear',
                        'shuffle_data' : True,
                        'data_transform' : data_transform,
                        'num_workers' : 8}


model_dict = {'psnr':
                {'2': 
                {'models':
                    {'0':'omarcos/deepopt/model-9r89t9j2:v0',
                     '1':'omarcos/deepopt/model-1nlkmche:v0',
                     '2':'omarcos/deepopt/model-lyt89k5t:v0',
                     '3':'omarcos/deepopt/model-2s01fb36:v0'
                    },
                    'order_0' : ['/home/obanmarcos/Balseiro/DeepOPT/datasets/x2/140519_5dpf_head_2', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x2/140117_3dpf_lower tail_2', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x2/140714_5dpf_head_2', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x2/140315_3dpf_head_2', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x2/140117_3dpf_head_2', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x2/140114_5dpf_head_2', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x2/140114_5dpf_upper tail_2', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x2/140117_3dpf_body_2', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x2/140315_1dpf_head_2', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x2/140117_3dpf_upper tail_2', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x2/140114_5dpf_body_2', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x2/140114_5dpf_lower tail_2']
                },
                '6':
                {'models':
                {'0':'omarcos/deepopt/model-bu7hlp3b:v0',
                     '1':'omarcos/deepopt/model-5sd2dysq:v0',
                     '2':'omarcos/deepopt/model-d58tvubf:v0',
                     '3':'omarcos/deepopt/model-1erq431o:v0'
                    },
                'order_0': ['/home/obanmarcos/Balseiro/DeepOPT/datasets/x6/140117_3dpf_lower tail_6', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x6/140714_5dpf_head_6', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x6/140114_5dpf_lower tail_6', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x6/140114_5dpf_upper tail_6', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x6/140117_3dpf_body_6', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x6/140519_5dpf_head_6', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x6/140117_3dpf_head_6', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x6/140315_3dpf_head_6', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x6/140114_5dpf_body_6','/home/obanmarcos/Balseiro/DeepOPT/datasets/x6/140315_1dpf_head_6', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x6/140117_3dpf_upper tail_6', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x6/140114_5dpf_head_6']
                },
                '10': 
                {'models':
                {'0':'omarcos/deepopt/model-203eopxg:v0',
                     '1':'omarcos/deepopt/model-oyxvdcj7:v0',
                     '2':'omarcos/deepopt/model-3fmp7pqn:v0',
                     '3':'omarcos/deepopt/model-zx2zo5x1:v0'},
                'order_0': ['/home/obanmarcos/Balseiro/DeepOPT/datasets/x10/140114_5dpf_upper tail_10', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x10/140315_3dpf_head_10', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x10/140117_3dpf_body_10', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x10/140315_1dpf_head_10', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x10/140117_3dpf_upper tail_10', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x10/140519_5dpf_head_10', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x10/140117_3dpf_head_10', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x10/140117_3dpf_lower tail_10', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x10/140114_5dpf_head_10', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x10/140714_5dpf_head_10', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x10/140114_5dpf_lower tail_10', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x10/140114_5dpf_body_10']
                },
                '14':
                {'models':
                {'0':'omarcos/deepopt/model-t96gkwcs:v0',
                     '1':'omarcos/deepopt/model-1jkmmgtf:v0',
                     '2':'omarcos/deepopt/model-f9j1q6r5:v0',
                     '3':'omarcos/deepopt/model-29x4g8yu:v0'
                    },
                'order_0': ['/home/obanmarcos/Balseiro/DeepOPT/datasets/x14/140117_3dpf_lower tail_14', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x14/140315_3dpf_head_14', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x14/140117_3dpf_head_14', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x14/140114_5dpf_head_14', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x14/140519_5dpf_head_14', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x14/140315_1dpf_head_14', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x14/140114_5dpf_lower tail_14', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x14/140714_5dpf_head_14', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x14/140117_3dpf_body_14', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x14/140114_5dpf_upper tail_14', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x14/140117_3dpf_upper tail_14', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x14/140114_5dpf_body_14']
                },
                '18':
                {'models':
                {'0': 'omarcos/deepopt/model-24gv33q2:v0',
                     '1': 'omarcos/deepopt/model-1qs0uo7v:v0',
                     '2':'omarcos/deepopt/model-2cpkr2dl:v0',
                     '3':'omarcos/deepopt/model-2cpkr2dl:v0'
                    },
                'order_0':['/home/obanmarcos/Balseiro/DeepOPT/datasets/x18/140315_3dpf_head_18', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x18/140117_3dpf_lower tail_18', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x18/140117_3dpf_upper tail_18', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x18/140117_3dpf_body_18', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x18/140114_5dpf_upper tail_18', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x18/140114_5dpf_body_18', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x18/140114_5dpf_head_18', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x18/140114_5dpf_lower tail_18', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x18/140519_5dpf_head_18', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x18/140714_5dpf_head_18', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x18/140315_1dpf_head_18', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x18/140117_3dpf_head_18']
                },
                '22':
                {'models':
                {'0': 'omarcos/deepopt/model-3dp1wex6:v0',
                     '1': 'omarcos/deepopt/model-2jwf0rwa:v0',
                     '2':'omarcos/deepopt/model-1qtf5f8u:v0',
                     '3':'omarcos/deepopt/model-2nxos558:v0'
                },
                'order_0':['/home/obanmarcos/Balseiro/DeepOPT/datasets/x22/140315_3dpf_head_22', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x22/140114_5dpf_head_22', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x22/140519_5dpf_head_22', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x22/140117_3dpf_body_22', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x22/140114_5dpf_upper tail_22', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x22/140315_1dpf_head_22', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x22/140114_5dpf_lower tail_22', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x22/140714_5dpf_head_22', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x22/140117_3dpf_head_22', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x22/140117_3dpf_lower tail_22', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x22/140117_3dpf_upper tail_22', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x22/140114_5dpf_body_22']
                },
                '26':
                {'models':
                {'0': 'omarcos/deepopt/model-32wj43mf:v0',
                     '1': 'omarcos/deepopt/model-3kmtjdm4:v0',
                     '2':'omarcos/deepopt/model-3l028zex:v0',
                     '3':'omarcos/deepopt/model-2jnmr8t0:v0'
                    },
                'order_0': ['/home/obanmarcos/Balseiro/DeepOPT/datasets/x26/140114_5dpf_head_26', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x26/140315_3dpf_head_26', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x26/140117_3dpf_head_26', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x26/140519_5dpf_head_26', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x26/140117_3dpf_lower tail_26', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x26/140114_5dpf_upper tail_26', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x26/140714_5dpf_head_26', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x26/140117_3dpf_body_26', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x26/140117_3dpf_upper tail_26', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x26/140114_5dpf_lower tail_26', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x26/140315_1dpf_head_26', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x26/140114_5dpf_body_26']
                    },
                },
            'ssim':
                {'2':
                {'models': 
                    {'0':'omarcos/deepopt/model-1knoqwz4:v0',
                     '1':'omarcos/deepopt/model-2r2yp6pi:v0',
                     '2':'omarcos/deepopt/model-2r6xowyu:v0',
                     '3':'omarcos/deepopt/model-10ocv8c8:v0'
                    },
                'order_0': ['/home/obanmarcos/Balseiro/DeepOPT/datasets/x2/140519_5dpf_head_2', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x2/140117_3dpf_lower tail_2', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x2/140714_5dpf_head_2', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x2/140315_3dpf_head_2', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x2/140117_3dpf_head_2', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x2/140114_5dpf_head_2', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x2/140114_5dpf_upper tail_2', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x2/140117_3dpf_body_2', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x2/140315_1dpf_head_2', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x2/140117_3dpf_upper tail_2', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x2/140114_5dpf_body_2', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x2/140114_5dpf_lower tail_2']
                },
                '6': 
                {'models': 
                {'0':'omarcos/deepopt/model-pt8a7a9u:v0',
                     '1':'omarcos/deepopt/model-2m5ccwz7:v0',
                     '2':'omarcos/deepopt/model-2jk5121n:v0',
                     '3':'omarcos/deepopt/model-334pu3db:v0'
                    },
                'order_0':['/home/obanmarcos/Balseiro/DeepOPT/datasets/x6/140117_3dpf_lower tail_6', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x6/140714_5dpf_head_6', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x6/140114_5dpf_lower tail_6', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x6/140114_5dpf_upper tail_6', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x6/140117_3dpf_body_6', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x6/140519_5dpf_head_6', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x6/140117_3dpf_head_6', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x6/140315_3dpf_head_6', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x6/140114_5dpf_body_6','/home/obanmarcos/Balseiro/DeepOPT/datasets/x6/140315_1dpf_head_6', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x6/140117_3dpf_upper tail_6', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x6/140114_5dpf_head_6']
                },
                '10':
                {'models':
                {'0':'omarcos/deepopt/model-2aoiqvuu:v0',
                     '1':'omarcos/deepopt/model-18qqq8ov:v0',
                     '2':'omarcos/deepopt/model-32yat3q5:v0',
                     '3':'omarcos/deepopt/model-1r53j8ys:v0'
                     },
                'order_0':['/home/obanmarcos/Balseiro/DeepOPT/datasets/x10/140114_5dpf_upper tail_10', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x10/140315_3dpf_head_10', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x10/140117_3dpf_body_10', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x10/140315_1dpf_head_10', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x10/140117_3dpf_upper tail_10', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x10/140519_5dpf_head_10', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x10/140117_3dpf_head_10', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x10/140117_3dpf_lower tail_10', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x10/140114_5dpf_head_10', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x10/140714_5dpf_head_10', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x10/140114_5dpf_lower tail_10', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x10/140114_5dpf_body_10']    
                },
                '14':
                {'models':
                {'0':'omarcos/deepopt/model-2sftdt8l:v0',
                     '1':'omarcos/deepopt/model-bp5tv3lf:v0',
                     '2':'omarcos/deepopt/model-33khl4s3:v0',
                     '3':'omarcos/deepopt/model-26eu06np:v0'
                    },
                'order_0':['/home/obanmarcos/Balseiro/DeepOPT/datasets/x14/140117_3dpf_lower tail_14', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x14/140315_3dpf_head_14', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x14/140117_3dpf_head_14', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x14/140114_5dpf_head_14', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x14/140519_5dpf_head_14', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x14/140315_1dpf_head_14', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x14/140114_5dpf_lower tail_14', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x14/140714_5dpf_head_14', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x14/140117_3dpf_body_14', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x14/140114_5dpf_upper tail_14', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x14/140117_3dpf_upper tail_14', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x14/140114_5dpf_body_14']
                },
                '18':
                {'models':
                {'0': 'omarcos/deepopt/model-39q68n8a:v0',
                     '1': 'omarcos/deepopt/model-1lorqlgs:v0',
                     '2':'omarcos/deepopt/model-3rkm7lhb:v0',
                     '3':'omarcos/deepopt/model-2e890ftb:v0'
                    },
                'order_0':['/home/obanmarcos/Balseiro/DeepOPT/datasets/x18/140315_3dpf_head_18', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x18/140117_3dpf_lower tail_18', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x18/140117_3dpf_upper tail_18', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x18/140117_3dpf_body_18', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x18/140114_5dpf_upper tail_18', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x18/140114_5dpf_body_18', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x18/140114_5dpf_head_18', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x18/140114_5dpf_lower tail_18', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x18/140519_5dpf_head_18', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x18/140714_5dpf_head_18', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x18/140315_1dpf_head_18', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x18/140117_3dpf_head_18']
                },
                '22':
                {'models':
                {'0': 'omarcos/deepopt/model-2srs5uf0:v0',
                     '1': 'omarcos/deepopt/model-sy28fr8u:v0',
                     '2':'omarcos/deepopt/model-3kx90j5f:v0',
                     '3':'omarcos/deepopt/model-nbhkzvx1:v0'
                    },
                'order_0':['/home/obanmarcos/Balseiro/DeepOPT/datasets/x22/140315_3dpf_head_22', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x22/140114_5dpf_head_22', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x22/140519_5dpf_head_22', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x22/140117_3dpf_body_22', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x22/140114_5dpf_upper tail_22', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x22/140315_1dpf_head_22', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x22/140114_5dpf_lower tail_22', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x22/140714_5dpf_head_22', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x22/140117_3dpf_head_22', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x22/140117_3dpf_lower tail_22', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x22/140117_3dpf_upper tail_22', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x22/140114_5dpf_body_22']
                },
                '26':
                {'models':
                {'0': 'omarcos/deepopt/model-2srs5uf0:v0',
                     '1': 'omarcos/deepopt/model-sy28fr8u:v0',
                     '2':'omarcos/deepopt/model-3kx90j5f:v0',
                     '3':'omarcos/deepopt/model-nbhkzvx1:v0'
                },
                'order_0':['/home/obanmarcos/Balseiro/DeepOPT/datasets/x26/140114_5dpf_head_26', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x26/140315_3dpf_head_26', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x26/140117_3dpf_head_26', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x26/140519_5dpf_head_26', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x26/140117_3dpf_lower tail_26', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x26/140114_5dpf_upper tail_26', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x26/140714_5dpf_head_26', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x26/140117_3dpf_body_26', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x26/140117_3dpf_upper tail_26', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x26/140114_5dpf_lower tail_26', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x26/140315_1dpf_head_26', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x26/140114_5dpf_body_26']
                }
                }
            }

def rotate_list(lst, n, direction = 'backwards'):
    '''
    Rotate list n-steps in direction 
    '''
    if direction == "backwards":
        for _ in range(0,n):
            lst.append(lst.pop(0))
    else: 
        for _ in range(0,n):
            lst.insert(0,lst.pop())
    return lst

def load_model(metric, fold, acceleration):

    run = wandb.init()
    print(metric, acceleration, 'models', fold)
    artifact = run.use_artifact(model_dict[metric][acceleration]['models'][fold], type='model')
    artifact_dir = artifact.download()
    
    model = MoDLReconstructor.load_from_checkpoint(Path(artifact_dir) / "model.ckpt", kw_dictionary_model_system = model_system_dict) 

    return model

# Corridas
def evaluate():
    
    accelerations = [str(i) for i in range(22,-2,-4)]
    foldings = [str(i) for i in range(4)]
    metrics = ['psnr']
    
    dataframe_path = 'logs/full_projections_dataframe.pkl'
    dict_path = 'logs/20-Dictionary_full_DeepOPT.pkl'
    
    dataframes_dict = {'psnr':{k:{} for k in accelerations}, 'ssim':{k:{} for k in accelerations}}
    model_system_dict['title'] = 'Projections'

    for metric in metrics:    

        for acceleration in accelerations:
            
            dataloader_dict['acceleration_factor'] = int(acceleration)
            model_system_dict['kw_dictionary_modl']['acceleration_factor'] = int(acceleration)
            modl_dict['metric'] = metric

            trainer_system = trutils.TrainerSystem(trainer_dict, dataloader_dict, model_system_dict)
            trainer = trainer_system.create_trainer()

            dataset_list = model_dict[metric][acceleration]['order_0']
            dataframe = pd.DataFrame(columns = ['test/psnr', 'test/ssim','test/psnr_admm', 'test/ssim_admm', 'test/psnr_fbp', 'test/ssim_fbp', 'test/psnr_unet', 'test/ssim_unet', 'fish_part', 'fish_dpf', 'datacode'])

            for fold in foldings:

                dataset_list_test = rotate_list(dataset_list, int(fold)*3)[-3:]
                
                model = load_model(metric, fold, acceleration)
                model.create_metrics()
                # model.load_unet('/home/obanmarcos/Balseiro/DeepOPT/saved_models/Unet_FA{}_Kfold{}.pth'.format(acceleration, fold))

                for folder in dataset_list_test:
                
                    test_dataloader = get_dataloader(folder, acceleration)

                    trainer.test(model = model, dataloaders = test_dataloader)

                    row = get_dataframe_row(model, folder, dataframe).copy()
                    dataframe = dataframe.append(row, ignore_index=True)
                    model.create_metrics()
                    
                    dataframes_dict[metric][acceleration] = dataframe    

                    with open(dict_path, 'wb') as f:

                        pickle.dump(dataframes_dict, f)


def get_dataframe_row(model, dataset_folder, dataframe):

    datacode = dataset_folder.split('_')[-4].split('/')[-1]
    fish_part = dataset_folder.split('_')[-2]
    fish_dpf = dataset_folder.split('_')[-3]

    row = {'test/psnr_admm': model.test_metric['test/psnr_admm'], 'test/ssim_admm': model.test_metric['test/ssim_admm'], 'test/psnr':model.test_metric['test/psnr'], 'test/ssim':model.test_metric['test/ssim'] ,'test/psnr_fbp':model.test_metric['test/psnr_fbp'], 'test/ssim_fbp': model.test_metric['test/ssim_fbp'], 'test/psnr_unet': model.test_metric['test/psnr_unet'], 'test/ssim_unet': model.test_metric['test/ssim_unet'], 'fish_part': fish_part, 'fish_dpf': fish_dpf, 'datacode':datacode}

    return row

def get_dataloader(dataset_folder, acceleration_factor):

    dataset_dict = {'root_folder' : dataset_folder, 
                    'acceleration_factor' : acceleration_factor,
                    'transform' : None}

    test_dataset = dlutils.ReconstructionDataset(**dataset_dict)    
    test_dataloader = DataLoader(test_dataset, 
                                batch_size = 1,
                                shuffle = False,
                                num_workers = 8)
    
    return test_dataloader


if __name__ == '__main__':

    # evaluate()
    add_deepopt_data()
