"""
Train the model with different K
author: obanmarcos
"""
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

# Using CPU or GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
folder_paths = [f140115_1dpf, f140315_3dpf, f140419_5dpf, f140714_5dpf, f140117_3dpf, f140114_5dpf] # Folders to be used

#%% Datasets 
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

train_name_modl = 'Optimization_K_PSNR_MODL_Test65'
train_name_SSIM = 'Optimization_K_SSIM_MODL_Test65'

test_models = False                 

tensor_path = datasets_folder+'Proj_{}_augmentFactor_{}_totalSize_{}_'.format(projection, augment_factor, total_size)
datasets = []
dataloaders = modutils.formDataloaders(datasets, projection, total_size, train_factor, val_factor, test_factor, batch_size, img_size, tensor_path, augment_factor, load_tensor = True, save_tensor = False)

nLayer = 8
lam = 0.05
max_angle = 720
Ks = np.arange(1, 11).astype(int)
epochs = 40
lr = 1e-4
                                       
test_loss_dict = {}

def train_networks():

    for K in Ks:   
        
        model_MODL_path = Path(model_folder+train_name_modl+'K_{}_lam_{}_nlay_{}_proj_{}.pth'.format(K, lam, nLayer, projection))
         
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
        
            with open(results_folder+train_name_modl+'Loss_K_Proj{}_nlay{}_epochs{}_K{}_lam{}_trnSize{}.pkl'.format(projection, nLayer, epochs, K, lam, train_factor), 'wb') as f:
        
                pickle.dump(train_infos, f)
                print('Diccionario salvado para proyección {}'.format(projection))
        
            modutils.save_net(model_folder+train_name_modl+'K_{}_lam_{}_nlay_{}_proj_{}'.format(K, lam, nLayer, projection), model_MODL)

        else:
            
            model_MODL = modl.OPTmodl(nLayer, K, max_angle, projection, img_size, None, lam, results_folder, shared = True, unet_options = False)
            modutils.load_net(model_folder+train_name_modl+'K_{}_lam_{}_nlay_{}_proj_{}'.format(K, lam, nLayer, projection), model_MODL, device)
        
        del model_MODL

        model_SSIM_path = Path(model_folder+train_name_SSIM+'K_{}_lam_{}_nlay_{}_proj_{}.pth'.format(K, lam, nLayer, projection))
        
        if not model_SSIM_path.is_file():                                            
             
            print('Training for MODL model (SSIM loss) with {} layers'.format(nLayer))
            loss_fn = SSIM(data_range = 1, size_average= True, channel = 1)
            loss_fbp_fn = SSIM(data_range = 1, size_average= True, channel = 1)
            loss_backproj_fn = SSIM(data_range = 1, size_average= True, channel = 1)

            # Training with SSIM as loss function 
            model_SSIM = modl.OPTmodl(nLayer, K, max_angle, projection, img_size, None, lam,  results_folder)
            optimizer_SSIM = torch.optim.Adam(model_SSIM.parameters(), lr = lr) 
            model_SSIM, train_info = modutils.model_training(model_SSIM, loss_fn, loss_backproj_fn, loss_fbp_fn, optimizer_SSIM, dataloaders, device, results_folder+train_name_SSIM, num_epochs = epochs, disp = True, do_checkpoint = 0, title = train_name_SSIM, plot_title = True, compute_ssim = True, compute_mse = True)

            with open(results_folder+train_name_SSIM+'Loss_K_SSIM_nlay{}_epochs{}_K{}_lam{}_trnSize{}.pkl'.format(nLayer, epochs, K, lam, train_factor), 'wb') as f:

                pickle.dump(train_info, f)
                print('Diccionario salvado para proyección {}'.format(projection))

            modutils.save_net(model_folder+train_name_SSIM+'K_{}_lam_{}_nlay_{}_proj_{}'.format(K, lam, nLayer, projection), model_SSIM)
        
        else:
            
            model_SSIM = modl.OPTmodl(nLayer, K, max_angle, projection, img_size, None, lam,  results_folder)
            modutils.load_net(model_folder+train_name_SSIM+'K_{}_lam_{}_nlay_{}_proj_{}'.format(K, lam, nLayer, projection), model_SSIM, device)
        
        del model_SSIM

def plot_testing():                                                                                                       
     with open(results_folder+train_name_modl+'K_SSIM_PSNR.pkl', 'rb') as f:
 
         test_loss_dict = pickle.load(f)
         
     fig, ax= plt.subplots(1,2, figsize = (12,6))
     col = ['red', 'blue', 'green']                                                                      
     alpha = 0.6
     capsize=5
     elinewidth = 2
     markeredgewidth= 2
     
     val_means = {k:[] for k in test_loss_dict[2].keys()}
 
     for K, losses in test_loss_dict.items():
         
         for i, (loss_key, val_loss) in enumerate(losses.items()):
         
             val_loss = np.array(val_loss)
             val_means[loss_key].append(val_loss.mean())
 
             if i<3:
 
                 if K==2:
                 
                     ax[0].errorbar(K, val_loss.mean(), yerr = val_loss.std(), marker = 'h', fmt = '-',c = col[i], label = loss_key, alpha = alpha, capsize=capsize, elinewidth=elinewidth, markeredgewidth=markeredgewidth)
                 
                 else:
         
                     ax[0].errorbar(K, val_loss.mean(), yerr = val_loss.std(), marker = 'h', fmt = '-',c = col[i], alpha = alpha, capsize=capsize, elinewidth=elinewidth, markeredgewidth=markeredgewidth)
              
     
             else:
 
                 i = i-3
                 
                 if K == 2:
                 
                     ax[1].errorbar(K, val_loss.mean(), yerr = val_loss.std(), marker = 'h', fmt = '-',c = col[i], label = loss_key, alpha = alpha, capsize=capsize, elinewidth=elinewidth, markeredgewidth=markeredgewidth)
                 
                 else:
                     
                     ax[1].errorbar(K, val_loss.mean(), yerr = val_loss.std(),marker = 'h', fmt = '-', c = col[i], alpha = alpha, capsize=capsize, elinewidth=elinewidth, markeredgewidth=markeredgewidth)
                 i = i+3
 
             if K == Ks[-1]:
                 
                 if i<3:
                     ax[0].plot(Ks, val_means[loss_key], c = col[i])
                
                 else:
     
                     ax[1].plot(Ks, val_means[loss_key], c = col[i-3])
 
     fig.suptitle('Acceleration factor X10') 

     ax[0].set_xlabel('Number of iterations (K)')
     ax[1].set_xlabel('Number of iterations (K)')
 
     ax[0].set_ylabel('PSNR in testing images')
     ax[1].set_ylabel('SSIM in testing images')
     handles, labels = ax[0].get_legend_handles_labels()
     
     labels_psnr = ['MODL w/PSNR loss', 'FBP', 'MODL w/SSIM loss']
 
     ax[0].legend(handles, labels_psnr)
     
     ax[0].grid(True)
     ax[1].grid(True)
 
     sns.despine(fig)
 
     fig.savefig(results_folder+'K_Comparison_SSIM_PSNR.pdf', bbox_inches = 'tight')

def acquire_testing():
    
    datasets = []

    tensor_path = datasets_folder+'Proj_{}_augmentFactor_{}_totalSize_{}_'.format(projection, augment_factor, total_size) 
    dataloaders = modutils.formDataloaders(datasets, projection, total_size, train_factor, val_factor, test_factor, batch_size, img_size, tensor_path, augment_factor, load_tensor = True, save_tensor = False)    
    
    for K in Ks:

        print('Load Model for {} projections'.format(projection))
        model_MODL = modl.OPTmodl(nLayer, K, max_angle, projection, img_size, None, lam, results_folder, shared = True, unet_options = False) 
        modutils.load_net(model_folder+train_name_modl+'K_{}_lam_{}_nlay_{}_proj_{}'.format(K, lam, nLayer, projection), model_MODL, device)

        model_SSIM = modl.OPTmodl(nLayer, K, max_angle, projection, img_size, None, lam, results_folder, True, useUnet = False)
        modutils.load_net(model_folder+train_name_SSIM+'K_{}_lam_{}_nlay_{}_proj_{}'.format(K, lam, nLayer, projection), model_SSIM, device)

        test_loss_fbp = []           
        test_loss_modl = []
        test_loss_ssim = []

        ssim_SSIM_test = []
        ssim_fbp_test = []
        ssim_modl_test = []
        
        loss_mse = torch.nn.MSELoss(reduction = 'sum')
       
        for inp, target, filt in tqdm(zip(dataloaders['test']['x'], dataloaders['test']['y'], dataloaders['test']['filtX'])): 
            
            pred_modl = model_MODL(inp)['dc'+str(K)]
            pred_ssim = model_SSIM(inp)['dc'+str(K)]

            loss_modl = loss_mse(pred_modl, target)
            loss_fbp = loss_mse(filt, target)
            loss_ssim = loss_mse(pred_ssim, target) 
        
            ssim_FBP = round(ssim(filt[0,0,...].detach().cpu().numpy(), target[0,0,...].detach().cpu().numpy()), 3)
            ssim_MODL = round(ssim(pred_modl[0,0,...].detach().cpu().numpy(),target[0,0,...].detach().cpu().numpy()), 3)
            ssim_SSIM = round(ssim(pred_ssim[0,0,...].detach().cpu().numpy(),target[0,0,...].detach().cpu().numpy()), 3)

            test_loss_ssim.append(modutils.psnr(img_size, loss_ssim.item(), 1))
            test_loss_modl.append(modutils.psnr(img_size, loss_modl.item(), 1))
            test_loss_fbp.append(modutils.psnr(img_size, loss_fbp.item(), 1))

            ssim_fbp_test.append(ssim_FBP)
            ssim_modl_test.append(ssim_MODL)
            ssim_SSIM_test.append(ssim_SSIM)

        test_loss_dict[K] = {'mse_modl': test_loss_modl, 'mse_fbp': test_loss_fbp, 'mse_ssim':test_loss_ssim, 'ssim_modl':ssim_modl_test, 'ssim_fbp':ssim_fbp_test, 'ssim_ssim':ssim_SSIM_test}    
        
        with open(results_folder+train_name_modl+'K_SSIM_PSNR.pkl', 'wb') as f:
            
            pickle.dump(test_loss_dict, f)
            print('Diccionario salvado para proyección {}'.format(projection))

if __name__ == '__main__':

    #train_networks()
    #acquire_testing()
    plot_testing()

