"""
Train the model with different number of layers
author: obanmarcos
"""
import os
import os,time, sys
os.chdir('/home/marcos/DeepOPT/')
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

# Using CPU or GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
folder_paths = [f140115_1dpf, f140315_3dpf, f140419_5dpf, f140714_5dpf] # Folders to be used

umbral_reg = 50

#%% Datasets 
# Training with more than one dataset
# Projnum == %10 of the data
proj_num = 72
nLayers = np.arange(3,10)

train_factor = 0.7
val_factor = 0.2
test_factor = 0.1                                                                                                                           
total_size = 3000                                                                                                                           
batch_size = 5                      
img_size = 100
augment_factor = 10                                             
# Load desired and undersampled datasets, on image space. Testing on Test Dataset
train_infos = {}                                                                                                                            
test_loss_dict = {}                                                                                                                                 tensor_path = datasets_folder+'Proj_{}_augmentFactor_{}_totalSize_{}_'.format(proj_num, augment_factor, total_size)                                                                                                                                                                         
datasets = modutils.formRegDatasets(folder_paths, umbral_reg, img_resize = img_size)                                                        
#datasets = []                                                                                                                              

dataloaders = modutils.formDataloaders(datasets, proj_num, total_size, train_factor, val_factor, test_factor, batch_size, img_size, tensor_path, augment_factor, load_tensor = False, save_tensor = True)    
train_name = 'Optimization_Layers_Test29'

lam = 0.05
max_angle = 640
K = 10
epochs = 50
lr = 5e-4

for nLayer in nLayers:   
    
    model = modl.OPTmodl(nLayer, K, max_angle, proj_num, img_size, None, lam, True, results_folder)
    loss_fn = torch.nn.MSELoss(reduction = 'sum')
    loss_fbp_fn = torch.nn.MSELoss(reduction = 'sum')
    loss_backproj_fn = torch.nn.MSELoss(reduction = 'sum')
    
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    
    model, train_info = modutils.model_training(model, loss_fn, loss_backproj_fn, loss_fbp_fn, optimizer, dataloaders, device, results_folder+train_name, num_epochs = epochs, disp = True, do_checkpoint = 0, title = train_name, plot_title = True)

    train_infos[K] = train_info
    
    print('Train MODL loss {}'.format(train_infos[K]['train'][-1]))
    print('Train FBP loss {}'.format(train_infos[K]['train_fbp'][-1]))

    #%% save loss for fbp and modl network
    with open(results_folder+train_name+'LossKs_nlay{}_epochs{}_K{}_lam{}_trnSize{}.pkl'.format(nLayer, epochs, K, lam, train_factor), 'wb') as f:
    
        pickle.dump(train_infos, f)
        print('Diccionario salvado para proyección {}'.format(proj_num))
    
    modutils.save_net(model_folder+train_name+'K_{}_lam_{}_nlay_{}_proj_{}'.format(K, lam, nLayer, proj_num), model)
    
    ### Testing part
    test_loss_total = []
    test_loss_fbp_total = []           
    test_loss_backproj_total = []

    for inp, target, filt in tqdm(zip(dataloaders['test']['x'], dataloaders['test']['y'], dataloaders['test']['filtX'])): 
        
        pred = model(inp)
        loss_test = loss_fn(pred['dc'+str(K)], target)
        loss_test_backproj = loss_backproj_fn(inp, target)
        loss_test_fbp = loss_fbp_fn(filt, target)
                                                                                                   
        test_loss_total.append(modutils.psnr(img_size, loss_test.item(), 1))
        test_loss_fbp_total.append(modutils.psnr(img_size, loss_test_fbp.item(), 1))
        test_loss_backproj_total.append(modutils.psnr(img_size, loss_test_backproj.item(), 1))
    
    modutils.plot_outputs(target, pred, results_folder+train_name+'Test_images_proj{}_K{}.pdf'.format(proj_num, K))

    test_loss_dict[nLayer] = {'loss_net': test_loss_total, 'loss_fbp': test_loss_fbp_total, 'loss_backproj':test_loss_backproj_total}
                                                                                                                
    with open(results_folder+train_name+'Layers_Proj{}_nLay{}_K{}_lam{}_trnSize{}.pkl'.format(proj_num, nLayer, K, lam, train_factor), 'wb') as f:
        
        pickle.dump(test_loss_dict, f)
        print('Diccionario salvado para proyección {}'.format(proj_num))
    
    del model, test_loss_total, test_loss_fbp_total
