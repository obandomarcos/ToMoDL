import numpy as np
import pickle 
import matplotlib.pyplot as plt
from utilities import dataloading_utilities as dlutils
from utilities.folders import *

from torch.utils.data import DataLoader
from training import train_utilities as trutils
import os, sys
from config import * 

sys.path.append(where_am_i())

from models.models_system import MoDLReconstructor
import wandb
from pathlib import Path

use_default_model_dict = True
use_default_dataloader_dict = True
use_default_trainer_dict = True

acceleration_factor = 20

metric = 'psnr'

if use_default_model_dict == True:
    # ResNet dictionary parameters
    resnet_options_dict = {'number_layers': 2,
                        'kernel_size':3,
                        'features':64,
                        'in_channels':1,
                        'out_channels':1,
                        'stride':1, 
                        'use_batch_norm': True,
                        'init_method': 'xavier'}

    # Model parameters
    modl_dict = {'use_torch_radon': False,
                'metric': metric,
                'number_layers': 2,
                'K_iterations' : 2,
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
    loss_dict = {'loss_name': metric,
                'psnr_loss': torch.nn.MSELoss(reduction = 'mean'),
                'ssim_loss': SSIM(data_range=1, size_average=True, channel=1),
                'msssim_loss': MSSSIM(kernel_size = 1)}

    # Optimizer parameters
    optimizer_dict = {'optimizer_name': 'Adam',
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
    # profiler = SimpleProfiler(dirpath = './logs/', filename = 'Test_training_profile_pytorch')
    # profiler = PyTorchProfiler(dirpath = './logs/', filename = 'Test_training_profile_pytorch')

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
    
    # data_transform = T.Compose([T.ToTensor()])
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
                        'batch_size' : 8, 
                        'sampling_method' : 'equispaced-linear',
                        'shuffle_data' : True,
                        'data_transform' : data_transform,
                        'num_workers' : 8}

run_name = 'CheckValues'

run = wandb.init(project = 'deepopt', reinit = True, job_type = 'Dataset Evaluation', name = run_name)
user_project_name = 'omarcos/deepopt/'
artifact_name = ''

artifact = run.use_artifact(user_project_name+artifact_name, type='model')
artifact_dir = artifact.download()

model = MoDLReconstructor.load_from_checkpoint(Path(artifact_dir) / "model.ckpt", kw_dictionary_model_system = model_system_dict) 

trainer_system = trutils.TrainerSystem(trainer_dict, dataloader_dict,model_system_dict)

dataset_list = ['140315_3dpf_head_20', '140114_5dpf_head_20', '140519_5dpf_head_20', '140117_3dpf_body_20', '140114_5dpf_upper tail_20', '140315_1dpf_head_20', '140114_5dpf_lower tail_20', '140714_5dpf_head_20', '140117_3dpf_head_20', '140117_3dpf_lower tail_20', '140117_3dpf_upper tail_20', '140114_5dpf_body_20']

test_dataset_folders = [datasets_folder+'x{}/'.format(acceleration_factor)+x for x in dataset_list[-3:]]

dataset_dict = {'root_folder' : test_dataset_folders[0], 
                            'acceleration_factor' : acceleration_factor,
                            'transform' : None}

test_dataset = dlutils.ReconstructionDataset(**dataset_dict)    

test_dataloader = DataLoader(test_dataset, 
                                        batch_size = 1,
                                        shuffle = False,
                                        num_workers = 8)


us_uf_im, us_uf_im
fig, ax = plt.subplots(1, 1, figsize = (8,6))

ax.imshow(, label = 'modl')
ax.plot(metrics['val_metric']['val/psnr_fbp'], label = 'fbp')

ax.legend()

fig.savefig('/home/obanmarcos/Balseiro/DeepOPT/results/15-CheckValues.pdf', bbox_inches = 'tight')