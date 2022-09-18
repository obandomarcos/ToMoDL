'''
PSNR methods check

author : obanmarcos
'''

import wandb
import torch
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

from torchvision import transforms as T
from pytorch_msssim import SSIM
from torchmetrics import MultiScaleStructuralSimilarityIndexMeasure as MSSSIM
from models.models_system import MoDLReconstructor
from pathlib import Path
from training import train_utilities as trutils

from skimage.metrics import structural_similarity as ssim 
from skimage.metrics import peak_signal_noise_ratio as psnr_skimage
from cv2 import PSNR as psnr_cv2

use_default_model_dict = True
use_default_dataloader_dict = True
use_default_trainer_dict = True
acceleration_factor = 22

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    modl_dict = {'use_torch_radon': False,
                'number_layers': 8,
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

    # Training parameters
    loss_dict = {'loss_name': 'psnr',
                'psnr_loss': torch.nn.MSELoss(reduction = 'mean'),
                'ssim_loss': SSIM(data_range=1, size_average=True, channel=1),
                'msssim_loss': MSSSIM(kernel_size = 1)}

    # Optimizer parameters
    optimizer_dict = {'optimizer_name': 'Adam+Tanh',
                    'lr': 1e-4}

    # System parameters
    model_system_dict = {'optimizer_dict': optimizer_dict,
                        'kw_dictionary_modl': modl_dict,
                        'loss_dict': loss_dict, 
                        'method':'unet',                       
                        'track_train': True,
                        'track_val': True,
                        'track_test': True,
                        'max_epochs':40}

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
                                'default_root_dir': model_folder}

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
    
    dataloader_dict = {'datasets_folder': datasets_folder,
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

artifact_names_x22_psnr = ['model-3dp1wex6:v0', 'model-2jwf0rwa:v0', 'model-1qtf5f8u:v0', 'model-2nxos558:v0']

dataset_list_x22 = ['140315_3dpf_head_22', '140114_5dpf_head_22', '140519_5dpf_head_22', '140117_3dpf_body_22', '140114_5dpf_upper tail_22', '140315_1dpf_head_22', '140114_5dpf_lower tail_22', '140714_5dpf_head_22', '140117_3dpf_head_22', '140117_3dpf_lower tail_22', '140117_3dpf_upper tail_22', '140114_5dpf_body_22']

def normalize_01(img):
    return (img-img.min())/(img.max()-img.min())

if __name__ == '__main__':
    
    part = dataset_list_x22[-3].split('_')[-2] 
    artifact_names = artifact_names_x22_psnr
    testing_name_group = 'x{}_test-PSNR'.format(acceleration_factor)

    run_name = 'test_metrics_kfold_x{}'.format(acceleration_factor)
    metric = 'psnr'
    dataset_list = dataset_list_x22 

    user_project_name = 'omarcos/deepopt/'

    trainer_system = trutils.TrainerSystem(trainer_dict, dataloader_dict, model_system_dict)
    trainer_system.set_datasets_list(dataset_list)
    
    train_dataloader, val_dataloader, test_dataloader = trainer_system.generate_K_folding_dataloader()

    artifact_dir = '/home/marcos/DeepOPT/artifacts/model-3dp1wex6:v0'
    
    model = MoDLReconstructor.load_from_checkpoint(Path(artifact_dir) / "model.ckpt", kw_dictionary_model_system = model_system_dict) 

    idx = 7
    batch_idx = 40

    for i, batch in enumerate(test_dataloader):
        
        unfiltered_us_rec, filtered_us_rec, filtered_fs_rec = batch
        
        if i == batch_idx:
            break

    unfiltered_us_rec_image = normalize_01(unfiltered_us_rec[idx,0,...].cpu().numpy())
    filtered_us_rec_image = normalize_01(filtered_us_rec[idx,0,...].cpu().numpy())
    filtered_fs_rec_image = normalize_01(filtered_fs_rec[idx,0,...].cpu().numpy())
    modl_reconstructed = normalize_01(model(unfiltered_us_rec.to(device))[idx,0,...].detach().cpu().numpy())

    images = [filtered_fs_rec_image, unfiltered_us_rec_image, filtered_us_rec_image,  modl_reconstructed]
    
    titles = ['Filtered backprojection - \nFully Sampled\n',
              'Unfiltered backprojection - \nUndersampled\n', 
              'Filtered backprojection - \nUndersampled X22\n',
              'MoDL reconstruction\n']

    metrics = ['',
               'SSIM: {}\n PSNR (CV2): {} dB\n PSNR (SKIMAGE): {} dB'.format(round(ssim(images[0], images[1]), 2), round(psnr_cv2(images[0], images[1], 1), 2), round(psnr_skimage(images[0], images[1]), 2)),
               'SSIM: {}\n PSNR (CV2): {} dB\n PSNR (SKIMAGE): {} dB'.format(round(ssim(images[0], images[2]), 2), round(psnr_cv2(images[0], images[2], 1), 2), round(psnr_skimage(images[0], images[2]), 2)),
               'SSIM: {}\n PSNR (CV2): {} dB\n PSNR (SKIMAGE): {}'.format(round(ssim(images[0], images[3]), 2), round(psnr_cv2(images[0], images[3], 1), 2), round(psnr_skimage(images[0], images[3]),2))]

    fig, axs = plt.subplots(2, len(images)//2, figsize = (8,10))
    axs = axs.flatten()
    
    for (ax, image, title, metric) in zip(axs, images, titles, metrics):

        ax.imshow(image)
        ax.set_title(title+metric)
        ax.set_axis_off()

    fig.savefig('./logs/Test_PSNR-11-{}.pdf'.format(part), bbox_inches = 'tight')

