'''
Definitive training for DeepOPT. The idea is to correct the repository issues from this script.
For this task, the steps were:
    * Device a training with the optimal DeepOPT parameters
        * K = 8, Layers = 8, init_lam = 0.05, max_angle = 720, K = 8, epochs = 40, lr = 1e-4
    * Train with K-Folding and check metrics for SSIM and PSNR losses  

Errors:
    * ENVIRONMENT VARIABLES            
'''

import os
import os,time, sys
os.chdir('/home/obanmarcos/Balseiro/DeepOPT/')
sys.path.append('Utilities/')
sys.path.append('OPTmodl/')
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import DataLoading as DL
from Folders_cluster import *
import ModelUtilities as modutils
import torch
from torch_radon import Radon, RadonFanbeam
from skimage.transform import radon, iradon
import phantominator as ph
import torchvision
import model_torch as modl
import pickle
from tqdm import tqdm
from pytorch_msssim import SSIM
from skimage.metrics import structural_similarity as ssim
from pathlib import Path
import seaborn as sns

def train_networks():
        
    # Using CPU or GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    folder_paths = [f140115_1dpf, f140315_3dpf, f140419_5dpf, f140714_5dpf, f140117_3dpf, f140114_5dpf] # Folders to be used

    # Datasets 
    # Training with more than one dataset
    # Projnum == %10 of the data
    train_factor = 0.7
    val_factor = 0.2
    test_factor = 0.1 
    total_size= 5000                 
    batch_size= 5 
    img_size = 100
    augment_factor = 1
    train_infos = {}        
    projection = 72
    k_iters = 6     # K_iters 
    k_fold_datasets = 2

    train_name_modl = 'KFolding_PSNR_MODL_Test71'
    train_name_SSIM = 'KFolding_SSIM_MODL_Test71'

    test_models = False                 
    tensor_path = None

    nLayer = 8
    lam = 0.05
    max_angle = 720
    K = 8
    epochs = 50
    lr = 1e-4
                                        
    test_loss_dict = {}

    datasets = modutils.formRegDatasets(folder_paths, fish_parts = None)
    
    for kfold in range(k_iters):
        
        datasets = modutils.k_fold_list(datasets, k_fold_datasets)
        
        dataloaders = modutils.formDataloaders(datasets, projection, total_size, train_factor, val_factor, test_factor, batch_size, img_size, tensor_path, augment_factor, load_tensor = False, save_tensor = False, k_fold_datasets = k_fold_datasets)

        model_MODL_path = Path(model_folder+train_name_modl+'K_{}_lam_{}_nlay_{}_proj_{}_kfold{}.pth'.format(K, lam, nLayer, projection, kfold))
        
        loss_mse = torch.nn.MSELoss(reduction = 'sum')
        
        if not model_MODL_path.is_file():
            print('Training for MODL model with {} layers'.format(nLayer))

            # Training MODL
            model_MODL = modl.OPTmodl(nLayer, K, max_angle, projection, img_size, None, lam, results_folder, shared = True, unet_options = False)

            loss_fn = torch.nn.MSELoss(reduction = 'sum')
            loss_fbp_fn = torch.nn.MSELoss(reduction = 'sum')
            loss_backproj_fn = torch.nn.MSELoss(reduction = 'sum') 

            optimizer_MODL = torch.optim.Adam(model_MODL.parameters(), lr = lr)
            
            model_MODL, train_info = modutils.model_training(model_MODL, loss_fn, loss_backproj_fn, loss_fbp_fn, optimizer_MODL, dataloaders, device, results_folder+train_name_modl, num_epochs = epochs, disp = True, do_checkpoint = 0, title = train_name_modl, plot_title = True, compute_mse = False, monai = False)

            train_infos[K] = train_info
        
            with open(results_folder+train_name_modl+'Loss_K_Proj{}_nlay{}_epochs{}_K{}_lam{}_trnSize{}_Kfold{}.pkl'.format(projection, nLayer, epochs, K, lam, train_factor, kfold), 'wb') as f:
        
                pickle.dump(train_infos, f)
                print('Diccionario salvado para proyección {}'.format(projection))
        
            modutils.save_net(model_folder+train_name_modl+'K_{}_lam_{}_nlay_{}_proj_{}__Kfold{}'.format(K, lam, nLayer, projection, kfold), model_MODL)

        else:
            
            model_MODL = modl.OPTmodl(nLayer, K, max_angle, projection, img_size, None, lam, results_folder, shared = True, unet_options = False)
            modutils.load_net(model_folder+train_name_modl+'K_{}_lam_{}_nlay_{}_proj_{}_Kfold{}'.format(K, lam, nLayer, projection, kfold), model_MODL, device)
        
        del model_MODL

        model_SSIM_path = Path(model_folder+train_name_SSIM+'K_{}_lam_{}_nlay_{}_proj_{}_Kfold{}.pth'.format(K, lam, nLayer, projection, kfold))
        
        if not model_SSIM_path.is_file():                                            
                
            print('Training for MODL model (SSIM loss) with {} layers'.format(nLayer))
            loss_fn = SSIM(data_range = 1, size_average= True, channel = 1)
            loss_fbp_fn = SSIM(data_range = 1, size_average= True, channel = 1)
            loss_backproj_fn = SSIM(data_range = 1, size_average= True, channel = 1)

            # Training with SSIM as loss function 
            model_SSIM = modl.OPTmodl(nLayer, K, max_angle, projection, img_size, None, lam,  results_folder)
            optimizer_SSIM = torch.optim.Adam(model_SSIM.parameters(), lr = lr) 
            model_SSIM, train_info = modutils.model_training(model_SSIM, loss_fn, loss_backproj_fn, loss_fbp_fn, optimizer_SSIM, dataloaders, device, results_folder+train_name_SSIM, num_epochs = epochs, disp = True, do_checkpoint = 0, title = train_name_SSIM, plot_title = True, compute_ssim = True, compute_mse = True)

            with open(results_folder+train_name_SSIM+'Loss_K_SSIM_nlay{}_epochs{}_K{}_lam{}_trnSize{}_Kfold{}.pkl'.format(nLayer, epochs, K, lam, train_factor, kfold), 'wb') as f:

                pickle.dump(train_info, f)
                print('Diccionario salvado para proyección {}'.format(projection))

            modutils.save_net(model_folder+train_name_SSIM+'K_{}_lam_{}_nlay_{}_proj_{}_Kfold{}'.format(K, lam, nLayer, projection, kfold), model_SSIM)
        
        else:
            
            model_SSIM = modl.OPTmodl(nLayer, K, max_angle, projection, img_size, None, lam,  results_folder)
            modutils.load_net(model_folder+train_name_SSIM+'K_{}_lam_{}_nlay_{}_proj_{}_Kfold{}'.format(K, lam, nLayer, projection, kfold), model_SSIM, device)
        
        del model_SSIM

if __name__ == '__main__':
    train_networks()