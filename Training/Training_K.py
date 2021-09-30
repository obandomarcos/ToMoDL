"""
Train the model with different K
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
train_dataset, test_dataset = modutils.formRegDatasets(folder_paths, umbral_reg)

#%% Datasets 
# Training with more than one dataset
# Projnum == %10 of the data
proj_num = 72
Ks = np.arange(5,11)

train_size = 1000
val_size = 200
test_size = 200
augment_factor = 3

batch_size = 5
img_size = 100

train_infos = {}
test_loss_dict = {}

for K in Ks:
    
    # Load desired and undersampled datasets, on image space. Testing on Test Dataset
    dataloaders = modutils.formDataloaders(train_dataset, test_dataset, proj_num, train_size, val_size, test_size, batch_size, img_size, augment_factor)
    
    #%% Model Settings
    nLayer = 3
    epochs = 40
    lam = 0.05
    max_angle = 720
    
    model = modl.OPTmodl(nLayer, K, proj_num, img_size, None, lam)
    loss_fn = torch.nn.MSELoss(reduction = 'sum')
    loss_fbp_fn = torch.nn.MSELoss(reduction = 'sum') 
    lr = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    
    model, train_info = modutils.model_training(model, loss_fn, loss_fbp_fn, optimizer, dataloaders, device, results_folder, num_epochs = epochs, disp = True, do_checkpoint = 0)
    train_infos[K] = train_info
    
    print('Train MODL loss {}'.format(train_infos[proj_num]['train'][-1]))
    print('Train FBP loss {}'.format(train_infos[proj_num]['train_fbp'][-1]))

    #%% save loss for fbp and modl network
    with open(results_folder+'LossKs_nlay{}_epochs{}_K{}_lam{}_trnSize{}.pkl'.format(nLayer, epochs, K, lam, train_size), 'wb') as f:
    
        pickle.dump(train_infos, f)
        print('Diccionario salvado para proyección {}'.format(proj_num))
    
    modutils.save_net(model_folder+'K_{}_lam_{}_nlay_{}_proj_{}'.format(K, lam, nLayer, proj_num), model)
    
    ### Testing part
    test_loss_total = []
    test_loss_fbp_total = []                                                                                                       
    for inp, target in tqdm(zip(dataloaders['test']['x'], dataloaders['test']['y'])): 
        
        pred = model(inp)
        loss_test = loss_fn(pred['dc'+str(K)], target)
        loss_test_fbp = loss_fbp_fn(inp, target)
                                                                                                   
        test_loss_total.append(modutils.psnr(img_size, loss_test.item()))
        test_loss_fbp_total.append(modutils.psnr(img_size, loss_test_fbp.item()))
    
    modutils.plot_outputs(target, pred, results_folder+'Test_images_proj{}_K{}.pdf'.format(proj_num, K))
    test_loss_dict[proj_num] = {'loss_net': test_loss_total, 'loss_fbp': test_loss_fbp_total}
                                                                                                                
    with open(results_folder+'KAnalisis_Proj{}_nLay{}_K{}_lam{}_trnSize{}.pkl'.format(proj_num, nLayer, K, lam, train_size), 'wb') as f:
        
        pickle.dump(test_loss_dict, f)
        print('Diccionario salvado para proyección {}'.format(proj_num))
    
    del model, test_loss_total, test_loss_fbp_total
