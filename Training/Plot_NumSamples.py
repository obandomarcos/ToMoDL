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
from pathlib import Path
import seaborn as sns
import matplotlib.animation as anim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
folder_paths = [f140115_1dpf, f140315_3dpf, f140419_5dpf, f140714_5dpf, f140117_3dpf, f140114_5dpf] # Folders to be used
    
projection = 72
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

train_name_modl = 'Optimization_NumberSamples_PSNR_MODL_Test63'
train_name_SSIM = 'Optimization_NumberSamples_SSIM_MODL_Test63'

test_loss_dict = {}

total_samples = [50, 100, 200, 500, 1000, 2000]

def acquire_testing():

    
    datasets = modutils.formRegDatasets(folder_paths, img_resize =img_size)
    num_samples_test = 5000

    tensor_path = datasets_folder+'Proj_{}_augmentFactor_{}_totalSize_{}_'.format(projection, augment_factor, num_samples_test) 
    dataloaders = modutils.formDataloaders(datasets, projection, num_samples_test, train_factor, val_factor, test_factor, batch_size, img_size, tensor_path, augment_factor, load_tensor = True, save_tensor = False)    

    for num_samples in total_samples:

        print('Load Model for {} projections'.format(projection))
        model_MODL = modl.OPTmodl(nLayer, K, max_angle, projection, img_size, None, lam, results_folder, shared = True, unet_options = False) 
        modutils.load_net(model_folder+train_name_modl+'K_{}_lam_{}_nlay_{}_proj_{}_numSamples{}'.format(K, lam, nLayer, projection, num_samples), model_MODL, device)

        model_SSIM = modl.OPTmodl(nLayer, K, max_angle, projection, img_size, None, lam, results_folder, True, useUnet = False)
        modutils.load_net(model_folder+train_name_SSIM+'K_{}_lam_{}_nlay_{}_proj_{}_numSamples{}'.format(K, lam, nLayer, projection, num_samples), model_SSIM, device)

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

        test_loss_dict[num_samples] = {'mse_modl': test_loss_modl, 'mse_fbp': test_loss_fbp, 'mse_ssim':test_loss_ssim, 'ssim_modl':ssim_modl_test, 'ssim_fbp':ssim_fbp_test, 'ssim_ssim':ssim_SSIM_test}    
        
        with open(results_folder+train_name_modl+'NumSamples_SSIM_PSNR.pkl', 'wb') as f:
            
            pickle.dump(test_loss_dict, f)
            print('Diccionario salvado para proyección {}'.format(projection))

def plot_testing():

    with open(results_folder+train_name_modl+'NumSamples_SSIM_PSNR.pkl', 'rb') as f:

        test_loss_dict = pickle.load(f)
        
    fig, ax= plt.subplots(1,2, figsize = (12,6))
    col = ['red', 'blue', 'green']                                                                      
    
    alpha = 0.6
    capsize=5
    elinewidth = 2
    markeredgewidth= 2
    
    val_means = {k:[] for k in test_loss_dict[50].keys()}

    for num_sample, losses in test_loss_dict.items():
        
        for i, (loss_key, val_loss) in enumerate(losses.items()):
        
            val_loss = np.array(val_loss)
            val_means[loss_key].append(val_loss.mean())

            if i<3:

                if num_sample==50:
                
                    ax[0].errorbar(num_sample, val_loss.mean(), yerr = val_loss.std(), marker = 'h', fmt = '-',c = col[i], label = loss_key, alpha = alpha, capsize=capsize, elinewidth=elinewidth, markeredgewidth=markeredgewidth)
                
                else:
        
                    ax[0].errorbar(num_sample, val_loss.mean(), yerr = val_loss.std(), marker = 'h', fmt = '-',c = col[i], alpha = alpha, capsize=capsize, elinewidth=elinewidth, markeredgewidth=markeredgewidth)
             
    
            else:

                i = i-3
                
                if num_sample == 50:
                
                    ax[1].errorbar(num_sample, val_loss.mean(), yerr = val_loss.std(), marker = 'h', fmt = '-',c = col[i], label = loss_key, alpha = alpha, capsize=capsize, elinewidth=elinewidth, markeredgewidth=markeredgewidth)
                
                else:
                    
                    ax[1].errorbar(num_sample, val_loss.mean(), yerr = val_loss.std(),marker = 'h', fmt = '-', c = col[i], alpha = alpha, capsize=capsize, elinewidth=elinewidth, markeredgewidth=markeredgewidth)
                i = i+3

            if num_sample == total_samples[-1]:
                
                if i<3:
                    ax[0].plot(total_samples, val_means[loss_key], c = col[i])
               
                else:
    
                    ax[1].plot(total_samples, val_means[loss_key], c = col[i-3])

    fig.suptitle('Acceleration factor X10') 
    

    ax[0].set_xscale('log') 
    ax[1].set_xscale('log')
    

    ax[0].set_xlabel('Number of training samples')
    ax[1].set_xlabel('Number of training samples')

    ax[0].set_ylabel('PSNR in testing images')
    ax[1].set_ylabel('SSIM in testing images')
    handles, labels = ax[0].get_legend_handles_labels()
    
    labels_psnr = ['MODL w/PSNR loss', 'FBP', 'MODL w/SSIM loss']

    ax[0].legend(handles, labels_psnr)
    
    ax[0].grid(True)
    ax[1].grid(True)

    sns.despine(fig)

    fig.savefig(results_folder+'NumSamples_Comparison_SSIM_PSNR.pdf', bbox_inches = 'tight')

def plot_images():
    
    datasets = modutils.formRegDatasets(folder_paths, img_resize =img_size)
    num_samples_test = 5000

    tensor_path = datasets_folder+'Proj_{}_augmentFactor_{}_totalSize_{}_'.format(projection, augment_factor, num_samples_test) 
    dataloaders = modutils.formDataloaders(datasets, projection, num_samples_test, train_factor, val_factor, test_factor, batch_size, img_size, tensor_path, augment_factor, load_tensor = True, save_tensor = False)    
    
    fig, ax = plt.subplots(2, len(total_samples), figsize = (12, 4))
    fig_fbp, ax_fbp = plt.subplots(1,1)
    
    imgs = {}

    inp = next(iter(dataloaders['test']['x']))
    target = next(iter(dataloaders['test']['y']))
    filt = next(iter(dataloaders['test']['filtX']))

    for j, num_samples in enumerate(total_samples):

        print('Load Model for {} projections'.format(projection))
        model_MODL = modl.OPTmodl(nLayer, K, max_angle, projection, img_size, None, lam, results_folder, shared = True, unet_options = False) 
        modutils.load_net(model_folder+train_name_modl+'K_{}_lam_{}_nlay_{}_proj_{}_numSamples{}'.format(K, lam, nLayer, projection, num_samples), model_MODL, device)

        model_SSIM = modl.OPTmodl(nLayer, K, max_angle, projection, img_size, None, lam, results_folder, True, useUnet = False)
        modutils.load_net(model_folder+train_name_SSIM+'K_{}_lam_{}_nlay_{}_proj_{}_numSamples{}'.format(K, lam, nLayer, projection, num_samples), model_SSIM, device)
       
        pred_modl = model_MODL(inp)
        pred_ssim = model_SSIM(inp)
        
        imgs_psnr = []
        imgs_ssim = []
        
        for (key1, img_psnr), (key2, img_ssim) in zip(pred_modl.items(), pred_ssim.items()):

            imgs_psnr.append(img_psnr.detach().cpu().numpy()[0,0,...])
            imgs_ssim.append(img_ssim.detach().cpu().numpy()[0,0,...])
            
        imgs[num_samples] = [imgs_psnr, imgs_ssim]
   
    ims_gif = []

    for k, key in enumerate(pred_modl.keys()):

        ims = [] 
        
        for j, num_samples in enumerate(total_samples):
            
#            ax[0][j].set_axis_off()
#            ax[1][j].set_axis_off()
                        
            ax[0][j].set_yticklabels([])
            ax[0][j].set_xticklabels([])

            ax[1][j].set_yticklabels([])
            ax[1][j].set_xticklabels([])

            ax[0][j].set_title('{} training\nsamples'.format(num_samples), fontsize = 8)
   
            ax[0][0].set_ylabel('PSNR loss')
            ax[1][0].set_ylabel('SSIM loss')

            im1 = ax[0][j].imshow(imgs[num_samples][0][k], cmap = 'gray')
            im2 = ax[1][j].imshow(imgs[num_samples][1][k], cmap = 'gray')
            
            if j == 0:
                
                txt1 = ax[0][j].text(10, 10, key, bbox={'facecolor': 'white', 'pad': 5}, fontsize = 10)                
                txt2 = ax[1][j].text(10, 10, key, bbox={'facecolor': 'white', 'pad': 5}, fontsize = 10)                
            
            ims = ims + [im1, im2, txt1, txt2]
        
        ims_gif.append(ims)
    
    ani = anim.ArtistAnimation(fig, ims_gif, interval=2000, blit=True,
                                repeat_delay=5000)
    
    ani.save(results_folder+'Training_Samples.gif', writer='imagemagick', fps=1)


if __name__ ==  '__main__':
    
    #acquire_testing()
    #plot_testing()
    plot_images()
