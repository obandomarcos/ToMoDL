"""
Train the model with different number of samples
author: obanmarcos
"""
import os
import os,time, sys
os.chdir('/home/marcos/DeepOPT/')
sys.path.append('utils')
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
folder_paths = [f140115_1dpf, f140315_3dpf, f140419_5dpf, f140714_5dpf, f140117_3dpf, f140114_5dpf] # Folders to be used

projection = 72
train_factor = 0.7
val_factor = 0.2
test_factor = 0.1
batch_size = 5 
img_size = 100
augment_factor = 1
train_infos = {}        

nLayer = 7
lam = 0.05
max_angle = 720
K = 8
epochs = 40
lr = 1e-4

shrink = 0.5
test_loss_dict = {} 

train_name_modl = 'Optimization_NumberSamples_PSNR_MODL_Test63'
train_name_SSIM = 'Optimization_FewShotLearning_SSIM_MODL_Test66'

test_models = True
train_MODL_PSNR = False

total_samples = [10, 20, 50]

datasets = modutils.formRegDatasets(folder_paths, img_resize =img_size)

for num_samples in total_samples:
   
    tensor_path = datasets_folder+'Proj_{}_augmentFactor_{}_totalSize_{}_'.format(projection, augment_factor, num_samples)

    dataloaders = modutils.formDataloaders(datasets, projection, num_samples, train_factor, val_factor, test_factor, batch_size, img_size, tensor_path, augment_factor, load_tensor = False, save_tensor = True)    
    model_MODL_path = Path(model_folder+train_name_modl+'K_{}_lam_{}_nlay_{}_proj_{}_numSamples{}.pth'.format(K, lam, nLayer, projection, num_samples))
     
    loss_mse = torch.nn.MSELoss(reduction = 'sum')
    
    if train_MODL_PSNR == True:

        if not model_MODL_path.is_file():
            print('Training for MODL model with {} projections'.format(projection))

            # Training MODL
            model_MODL = modl.OPTmodl(nLayer, K, max_angle, projection, img_size, None, lam, results_folder, shared = True, unet_options = False)

            loss_fn = torch.nn.MSELoss(reduction = 'sum')
            loss_fbp_fn = torch.nn.MSELoss(reduction = 'sum')
            loss_backproj_fn = torch.nn.MSELoss(reduction = 'sum') 

            optimizer_MODL = torch.optim.Adam(model_MODL.parameters(), lr = lr)
            
            model_MODL, train_info = modutils.model_training(model_MODL, loss_fn, loss_backproj_fn, loss_fbp_fn, optimizer_MODL, dataloaders, device, results_folder+train_name_modl, num_epochs = epochs, disp = True, do_checkpoint = 0, title = train_name_modl, plot_title = True, compute_mse = False, monai = False)

            train_infos[projection] = train_info
        
            with open(results_folder+train_name_modl+'LossSamples_Proj{}_nlay{}_epochs{}_K{}_lam{}_trnSize{}.pkl'.format(projection, nLayer, epochs, K, lam, train_factor), 'wb') as f:
        
                pickle.dump(train_infos, f)
                print('Diccionario salvado para proyección {}'.format(projection))
        
            modutils.save_net(model_folder+train_name_modl+'K_{}_lam_{}_nlay_{}_proj_{}_numSamples{}'.format(K, lam, nLayer, projection, num_samples), model_MODL)

        else:
            
            model_MODL = modl.OPTmodl(nLayer, K, max_angle, projection, img_size, None, lam, results_folder, shared = True, unet_options = False)
            modutils.load_net(model_folder+train_name_modl+'K_{}_lam_{}_nlay_{}_proj_{}_numSamples{}'.format(K, lam, nLayer, projection, num_samples), model_MODL, device)
        
        del model_MODL

    model_SSIM_path = Path(model_folder+train_name_SSIM+'K_{}_lam_{}_nlay_{}_proj_{}_numSamples{}.pth'.format(K, lam, nLayer, projection, num_samples))
    
    if not model_SSIM_path.is_file():                                            
         
        print('Training for MODL model with {} projections'.format(projection))
        loss_fn = SSIM(data_range = 1, size_average= True, channel = 1)
        loss_fbp_fn = SSIM(data_range = 1, size_average= True, channel = 1)
        loss_backproj_fn = SSIM(data_range = 1, size_average= True, channel = 1)

        # Training with SSIM as loss function 
        model_SSIM = modl.OPTmodl(nLayer, K, max_angle, projection, img_size, None, lam,  results_folder)
        optimizer_SSIM = torch.optim.Adam(model_SSIM.parameters(), lr = lr) 
        model_SSIM, train_info = modutils.model_training(model_SSIM, loss_fn, loss_backproj_fn, loss_fbp_fn, optimizer_SSIM, dataloaders, device, results_folder+train_name_SSIM, num_epochs = epochs, disp = True, do_checkpoint = 0, title = train_name_SSIM, plot_title = True, compute_ssim = True, compute_mse = True)

        with open(results_folder+train_name_SSIM+'LossSSIM_nlay{}_epochs{}_K{}_lam{}_trnSize{}.pkl'.format(nLayer, epochs, K, lam, train_factor), 'wb') as f:

            pickle.dump(train_info, f)
            print('Diccionario salvado para proyección {}'.format(projection))

        modutils.save_net(model_folder+train_name_SSIM+'K_{}_lam_{}_nlay_{}_proj_{}_numSamples{}'.format(K, lam, nLayer, projection, num_samples), model_SSIM)
    
    else:
        
        model_SSIM = modl.OPTmodl(nLayer, K, max_angle, projection, img_size, None, lam,  results_folder)
        modutils.load_net(model_folder+train_name_SSIM+'K_{}_lam_{}_nlay_{}_proj_{}_numSamples{}'.format(K, lam, nLayer, projection, num_samples), model_SSIM, device)
    
    with torch.no_grad(): 

        if test_models == True:
            num_samples = 5000 
            
            train_factor = 0.0
            val_factor = 0.0
            test_factor = 1.0

            tensor_path = datasets_folder+'Proj_{}_augmentFactor_{}_totalSize_{}_'.format(projection, augment_factor, num_samples)                              
            dataloaders = modutils.formDataloaders(datasets, projection, num_samples, train_factor, val_factor, test_factor, batch_size, img_size, tensor_path, augment_factor, load_tensor = True, save_tensor = False)    

            mse_loss_modl = []
            mse_loss_fbp = []
            mse_loss_modlssim = []

            ssim_loss_modl = []
            ssim_loss_fbp = []
            ssim_loss_modlssim = []

            for inp, target, filt in tqdm(zip(dataloaders['test']['x'], dataloaders['test']['y'], dataloaders['test']['filtX'])): 
                
                if train_MODL_PSNR == True:
                    pred_modl = model_MODL(inp) 
                    loss_modl_MSE = loss_mse(pred_modl['dc'+str(K)], target)
                    mse_loss_modl.append(modutils.psnr(img_size, loss_modl_MSE.item(), 1))
                    loss_modl_SSIM = ssim(pred_modl['dc'+str(K)].detach().cpu().numpy()[0,0,...], target.detach().cpu().numpy()[0,0,...])
                    ssim_loss_modl.append(loss_modl_SSIM)

                pred_ssim = model_SSIM(inp)
                
                loss_modlSSIM_MSE = loss_mse(pred_ssim['dc'+str(K)], target)
                loss_fbp_MSE = loss_mse(filt, target)
                                                                     
                mse_loss_fbp.append(modutils.psnr(img_size, loss_fbp_MSE.item(), 1))
                mse_loss_modlssim.append(modutils.psnr(img_size, loss_modlSSIM_MSE.item(), 1))
            
                loss_modlSSIM_SSIM = ssim(pred_ssim['dc'+str(K)].detach().cpu().numpy()[0,0,...], target.detach().cpu().numpy()[0,0,...])
                loss_fbp_SSIM = ssim(filt.detach().cpu().numpy()[0,0,...], target.detach().cpu().numpy()[0,0,...])                               

                ssim_loss_fbp.append(loss_modlSSIM_MSE)
                ssim_loss_modlssim.append(loss_fbp_SSIM)
            
            test_loss_dict[projection] = {'mse_modl': mse_loss_modl, 'mse_modlssim':mse_loss_modlssim, 'mse_fbp': mse_loss_fbp, 'ssim_modl':ssim_loss_modl, 'ssim_modlssim':ssim_loss_modlssim, 'ssim_fbp':ssim_loss_fbp}

            with open(results_folder+train_name_SSIM+'NumSamples_Proj{}_nLay{}_K{}_lam{}_trnSize{}_NumSamples_{}.pkl'.format(projection, nLayer, K, lam, train_factor, num_samples), 'wb') as f:
            
                pickle.dump(test_loss_dict, f)
                print('Diccionario salvado para proyección {}'.format(projection))
        
