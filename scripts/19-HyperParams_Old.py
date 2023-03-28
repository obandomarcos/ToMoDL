"""
Test MODL trained with PSNR and SSIM loss with different acceleration factors
"""
#%% Import libraries
import os
import os,time, sys
from config import * 
import wandb

os.chdir('../DeepOPT-2/')
sys.path.append('Utilities/')
sys.path.append('OPTmodl/')
sys.path.append('Reconstruction/')

sys.path.append(where_am_i())

from utilities import dataloading_utilities as dlutils
from utilities.folders import *
from models.models_system import MoDLReconstructor
from pathlib import Path
from training import train_utilities as trutils

import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt
import DataLoading as DL
from Folders_cluster import *
import Reconstruction as RecTV
import ModelUtilities as modutils
import torch
from torch_radon import Radon, RadonFanbeam
from skimage.transform import radon, iradon
import torchvision
import model_torch as modl
import pickle
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
import seaborn as sns

from torchvision import transforms as T
from pytorch_msssim import SSIM
from torchmetrics import MultiScaleStructuralSimilarityIndexMeasure as MSSSIM

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

factors = np.arange(2,20,2)
number_projections = 720//factors

train_factor = 0.7
val_factor = 0.2
test_factor = 0.1 
total_size = 5000                  
batch_size = 5 
img_size = 100
augment_factor = 1

#%% Model Settings

train_name_modl = 'Optimization_K_SSIM_MODL_Test65'

test_loss_dict = {}

metric = 'ssim'
acceleration_factor = 10

use_default_model_dict = True
use_default_dataloader_dict = True
use_default_trainer_dict = True

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
                'metric': metric,
                'K_iterations' : 4,
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
                        'track_alternating_admm': True,
                        'admm_dictionary': admm_dictionary,         
                        'track_alternating_twist': True,
                        'twist_dictionary': twist_dictionary}

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

dataset_list = ['140315_3dpf_head_10', '140114_5dpf_head_10', '140519_5dpf_head_10', '140117_3dpf_body_10', '140114_5dpf_upper tail_10', '140315_1dpf_head_10', '140114_5dpf_lower tail_10', '140117_3dpf_lower tail_10', '140117_3dpf_upper tail_10','140117_3dpf_head_10', '140714_5dpf_head_10', '140114_5dpf_body_10']

def acquire_testing():
    
    print('Load Model for {} projections'.format(modl_dict['number_projections_undersampled']))
    model_MODL = modl.OPTmodl(resnet_options_dict['number_layers'], modl_dict['K_iterations'], modl_dict['number_projections_total'], modl_dict['number_projections_undersampled'], modl_dict['image_size'], None, modl_dict['lambda'], results_folder, shared = True, unet_options = False) 
    
    modutils.load_net(model_folder+train_name_modl+'K_{}_lam_{}_nlay_{}_proj_{}'.format(modl_dict['K_iterations'], modl_dict['lambda'], resnet_options_dict['number_layers'], modl_dict['number_projections_undersampled']), model_MODL, device)    

    return model_MODL

def log_plot(target, prediction, path):
        '''
        Plots target and prediction (unrolled) and logs it. 
        '''
        
        fig, ax = plt.subplots(1, len(prediction.keys())+1, figsize = (16,6))

        im = ax[0].imshow(target.clone().detach().cpu().numpy()[0,0,:,:], cmap = 'gray')
        ax[0].set_title('Target')
        ax[0].axis('off') 
        
        # plt.suptitle('Epoch {} in {} phase'.format(self.current_epoch, phase))

        for a, (key, image) in zip(ax[1:], prediction.items()):

            im = a.imshow(image.clone().detach().cpu().numpy()[0,0,:,:], cmap = 'gray')
            a.set_title(key)
            a.axis('off')
        
        cax = fig.add_axes([a.get_position().x1+0.01,a.get_position().y0,0.02,a.get_position().height])
        plt.colorbar(im, cax = cax)

        plt.savefig(path, bbox_inches = 'tight')
        #wandb.log({'{}_plot_{}'.format(phase, self.current_epoch): fig})
        plt.close(fig)

if __name__ == '__main__':
    
    wandb.init(mode = 'offline')
    model = acquire_testing()
    lightning_model = MoDLReconstructor(model_system_dict)
    
    lightning_model.model = model

    trainer_system = trutils.TrainerSystem(trainer_dict, dataloader_dict, model_system_dict)
    # trainer = trainer_system.create_trainer()

    train_dataloader, val_dataloader, test_dataloader = trainer_system.generate_K_folding_dataloader()

    for i, (us_uf_image, us_fil_image, fs_fil_image) in enumerate(test_dataloader):
        
        us_uf_image = us_uf_image.to(device)
        us_fil_image = us_fil_image.to(device) 
        fs_fil_image = fs_fil_image.to(device)

        deepopt_image = model(us_uf_image)
        log_plot(fs_fil_image, deepopt_image, '/home/obanmarcos/Balseiro/DeepOPT/results/19-HyperparamsOld_Transfer.pdf')
        deepopt_image = deepopt_image['dc'+str(model.K)]
        
        ssim_loss_deepopt = loss_dict['ssim_loss'](fs_fil_image, deepopt_image).detach().cpu().numpy()
        psnr_loss_deepopt = 10*np.log10(1/loss_dict['psnr_loss'](fs_fil_image, deepopt_image).detach().cpu().numpy())

        ssim_loss_fbp = loss_dict['ssim_loss'](fs_fil_image, us_uf_image).detach().cpu().numpy()
        psnr_loss_fbp = 10*np.log10(1/loss_dict['psnr_loss'](fs_fil_image, us_uf_image).detach().cpu().numpy())
        
        print(ssim_loss_fbp, ssim_loss_deepopt) 
        
        if i == 5:

            break 
