"""
Test MODL trained with PSNR and SSIM loss with different acceleration factors
"""
#%% Import libraries
import os
import os,time, sys
os.chdir('.')
sys.path.append('Utilities/')
sys.path.append('OPTmodl/')
sys.path.append('Reconstruction/')

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
from bayes_opt import BayesianOptimization
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.metrics import structural_similarity as ssim
import seaborn as sns

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
shrink = 0.5
nLayer = 7
K = 8
epochs = 1
lam = 0.05
max_angle = 720
lr = 0.001

K_SSIM = 7
nLayer_SSIM = 8
proj_num = 72
lam_SSIM = 0.001

train_name_SSIM = 'Optimization_Projections_SSIM_MODL_Test62'
train_name_modl = 'Optimization_Projections_PSNR_MODL_Test62'


test_loss_dict = {}

def acquire_testing():

    for projection, factor in zip(number_projections, factors):

        print('Load dataset for {} projections'.format(projection))
        tensor_path = datasets_folder+'Proj_{}_augmentFactor_{}_totalSize_{}_'.format(projection, augment_factor, total_size)
        datasets = []
        dataloaders = modutils.formDataloaders(datasets, proj_num, total_size, train_factor, val_factor, test_factor, batch_size, img_size, tensor_path, augment_factor, load_tensor = True, save_tensor = False)    

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

        test_loss_dict[projection] = {'mse_modl': test_loss_modl, 'mse_fbp': test_loss_fbp, 'mse_ssim':test_loss_ssim, 'ssim_modl':ssim_modl_test, 'ssim_fbp':ssim_fbp_test, 'ssim_ssim':ssim_SSIM_test}    
        
        with open(results_folder+train_name_modl+'Projections_SSIM_PSNR.pkl', 'wb') as f:
            
            pickle.dump(test_loss_dict, f)
            print('Diccionario salvado para proyección {}'.format(projection))

def plot_images():

    with open(results_folder+train_name_modl+'Projections_SSIM_PSNR.pkl', 'rb') as f:
        test_loss_dict = pickle.load(f)
        print('Diccionario cargado para proyecciones')

    fig, ax= plt.subplots(1,2, figsize = (12,6))
    col = ['red', 'blue', 'green']                                                                      
    
    alpha = 0.6
    capsize=5
    elinewidth = 2
    markeredgewidth= 2
    
    val_means = {k:[] for k in test_loss_dict[180].keys()}                                                                                                                                                                                                                                 
    for factor, (projection, losses) in zip(factors, test_loss_dict.items()):
        
        for i, (loss_key, val_loss) in enumerate(losses.items()):
             
            val_loss = np.array(val_loss)
            val_means[loss_key].append(val_loss.mean())                                                                                                                                                                             
            if i<3:                                                                                                                                                                                                              
                if projection ==180:
                
                    ax[0].errorbar(factor, val_loss.mean(), yerr = val_loss.std(), marker = 'h', fmt = '-',c = col[i], label = loss_key, alpha = alpha, capsize=capsize, elinewidth=elinewidth, markeredgewidth=markeredgewidth)
                
                else:
        
                    ax[0].errorbar(factor, val_loss.mean(), yerr = val_loss.std(), marker = 'h', fmt = '-',c = col[i], alpha = alpha, capsize=capsize, elinewidth=elinewidth, markeredgewidth=markeredgewidth)
             
    
            else:                                                                                                                                                                                        
                i = i-3
                
                if projection == 180:
                
                    ax[1].errorbar(factor, val_loss.mean(), yerr = val_loss.std(), marker = 'h', fmt = '-',c = col[i], label = loss_key, alpha = alpha, capsize=capsize, elinewidth=elinewidth, markeredgewidth=markeredgewidth)
                
                else:
                    
                    ax[1].errorbar(factor, val_loss.mean(), yerr = val_loss.std(),marker = 'h', fmt = '-', c = col[i], alpha = alpha, capsize=capsize, elinewidth=elinewidth, markeredgewidth=markeredgewidth)
                i = i+3
                                                                                                                                                                                                                                     
            if projection == number_projections[-1]:
                
                if i<3:
                    ax[0].plot(factors, val_means[loss_key], c = col[i])
               
                else:
    
                    ax[1].plot(factors, val_means[loss_key], c = col[i-3])
                                                                                                                                                                                                                                     
    ax[0].set_xlabel('Acceleration factor')
    ax[1].set_xlabel('Acceleration factor')
                                                                                                                                                                                                        
    ax[0].set_ylabel('PSNR in testing images')
    ax[1].set_ylabel('SSIM in testing images')
    handles, labels = ax[0].get_legend_handles_labels()
    
    labels_psnr = ['MODL w/PSNR loss', 'FBP', 'MODL w/SSIM loss']                                                                                                                                                                                                                         
    ax[0].legend(handles, labels_psnr)
    
    ax[0].grid(True)
    ax[1].grid(True)

    sns.despine(fig)                                                                                                                                                                                  
    fig.savefig(results_folder+'Projections_Comparison_SSIM_PSNR.pdf', bbox_inches = 'tight')

if __name__ ==  '__main__':
    
    acquire_testing()
    plot_images()
