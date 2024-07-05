'''
PSNR methods check

author : obanmarcos
'''

from cgi import test
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
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure as MSSSIM
from models.models_system import MoDLReconstructor
from models.modl import MoDL
from pathlib import Path
from training import train_utilities as trutils

from skimage.metrics import structural_similarity as ssim 
from skimage.metrics import peak_signal_noise_ratio as psnr_skimage
from cv2 import PSNR as psnr_cv2
from matplotlib.patches import Rectangle

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
                        'verbose':1,
                        'initialization':0,
                        'max_iter':10000, 
                        'gpu':0,
                        'tau': 0.02}
    
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
                        'track_alternating_admm': True,
                        'admm_dictionary': admm_dictionary,
                        'track_alternating_twist': True,
                        'twist_dictionary': twist_dictionary,
                        'track_unet': True,
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
                        'use_subset_by_part': False,
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
                        'num_workers' : 16}

artifact_names_psnr = ['model-3dp1wex6:v0', 'model-2jwf0rwa:v0', 'model-1qtf5f8u:v0', 'model-2nxos558:v0']

dataset_list = ['/home/obanmarcos/Balseiro/DeepOPT/datasets/x22/140114_5dpf_head_22', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x22/140519_5dpf_head_22', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x22/140117_3dpf_body_22', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x22/140114_5dpf_upper tail_22', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x22/140315_1dpf_head_22', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x22/140114_5dpf_lower tail_22', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x22/140714_5dpf_head_22', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x22/140117_3dpf_head_22', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x22/140117_3dpf_lower tail_22', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x22/140315_3dpf_head_22', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x22/140117_3dpf_upper tail_22', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x22/140114_5dpf_body_22']

def normalize_01(img):

    return (img-img.min())/(img.max()-img.min())


def psnr_per_box(target, rec, box):

    target_region = target[box[0,0]:box[0,1],box[1,0]:box[1,1]]
    rec_region = rec[box[0,0]:box[0,1],box[1,0]:box[1,1]] 

    target_region = (target_region-target_region.mean())/target_region.std()
    rec_region = (rec_region-rec_region.mean())/rec_region.std()
    im_range = target_region.max()-target_region.min()

    return round(psnr_skimage(target_region, rec_region, data_range = im_range), 2)


def ssim_per_box(target, rec, box):

    target_region = target[box[0,0]:box[0,1],box[1,0]:box[1,1]]
    rec_region = rec[box[0,0]:box[0,1],box[1,0]:box[1,1]] 

    target_region = (target_region-target_region.mean())/target_region.std()
    rec_region = (rec_region-rec_region.mean())/rec_region.std()
    im_range = target_region.max()-target_region.min()

    return round(ssim(target_region, rec_region, data_range = im_range), 2)

def mean_box(image, c, w, h):
    
    image = image[c[0]:c[0]+w, c[1]:c[1]+h]
    image = (image-image.mean())/image.std()
    
    return np.round(np.mean(image), 2)

def box_list(c, h, w):

    return np.array([[c[0], c[0]+w], [c[1], c[1]+h]])

def normalize_std(img):

    return (img-img.mean())/img.std()


if __name__ == '__main__':
    
    part = dataset_list[-3].split('_')[-2] 
    artifact_names = artifact_names_psnr
    testing_name_group = 'x{}_test-PSNR'.format(acceleration_factor)

    run_name = 'test_metrics_kfold_x{}'.format(acceleration_factor)
    metric = 'psnr'

    user_project_name = 'omarcos/deepopt/'

    trainer_system = trutils.TrainerSystem(trainer_dict, dataloader_dict, model_system_dict)
    trainer_system.set_datasets_list(dataset_list)
    
    _, val_dataloader, test_dataloader = trainer_system.generate_K_folding_dataloader()

    artifact_dir = '/home/obanmarcos/Balseiro/DeepOPT/artifacts/{}'.format(artifact_names_psnr[0])
    
    model = MoDLReconstructor.load_from_checkpoint(Path(artifact_dir) / "model.ckpt", kw_dictionary_model_system = model_system_dict)

    # trainer = trainer_system.create_trainer()
    # wandb.init()
    # trainer.test(model, test_dataloader)

    idx = 1
    batch_idx = 58

    for i, batch in enumerate(test_dataloader):
        
        unfiltered_us_rec, filtered_us_rec, filtered_fs_rec = batch
        
        if i == batch_idx:
            break
    
    del test_dataloader

    unfiltered_us_rec_image = unfiltered_us_rec[idx,0,...].cpu().numpy()
    filtered_us_rec_image =  normalize_01(filtered_us_rec[idx,0,...].cpu().numpy())
    target = filtered_fs_rec
    prediction = model.model(unfiltered_us_rec.to(device))

    
    target = filtered_fs_rec
    # model.model.lam = torch.nn.Parameter(torch.tensor(torch.FloatTensor([lam]), device = device))
    # prediction = model.model(unfiltered_us_rec.to(device))
    
    phase = 'train'
    fig, ax = plt.subplots(1, len(prediction.keys())+1, figsize = (16,6))
    
    im = ax[0].imshow(target.detach().cpu().numpy()[0,0,:,:], cmap = 'gray')
    ax[0].set_title('Target')
    ax[0].axis('off') 
    
    plt.suptitle('Epoch {} in {} phase'.format(37, phase))

    for a, (key, image) in zip(ax[1:], prediction.items()):

        im = a.imshow(image.detach().cpu().numpy()[0,0,:,:], cmap = 'gray')
        a.set_title(key)
        a.axis('off')
    
    cax = fig.add_axes([a.get_position().x1+0.01,a.get_position().y0,0.02,a.get_position().height])
    plt.colorbar(im, cax = cax)
    
    target = target.detach().cpu().numpy()[0,0,:,:]
    pred = prediction['dc'+str(8)].detach().cpu().numpy()[0,0,:,:]
    
    ssim_images = ssim(normalize_01(target), normalize_01(pred))
    ssim_real = ssim(normalize_01(target), filtered_us_rec_image)

    psnr_images = psnr_skimage(normalize_std(target), normalize_std(pred), data_range = normalize_std(target).max().item()-normalize_std(target).min().item())
    psnr_real = psnr_skimage(normalize_std(target), filtered_us_rec_image, data_range = normalize_std(target).max().item()-normalize_std(target).min().item())
    
    print('Modelo SSIM: {}, psnr {}'.format(ssim_images, psnr_images))
    print('FBP: {}, psnr {}'.format(ssim_real, psnr_real))
    
    fig.savefig('/home/obanmarcos/Balseiro/DeepOPT/results/lambdas/Check_newmodel_problem_torchradon.png', bbox_inches = 'tight')

    plt.close(fig)
