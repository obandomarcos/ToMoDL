"""
Train the model with different number of projections
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
folder_paths = [f140115_1dpf, f140315_3dpf, f140419_5dpf, f140714_5dpf, f140117_3dpf, f140114_5dpf] # Folders to be used

proj_num = 720//np.arange(2,26,4)
train_factor = 0.7
val_factor = 0.2
test_factor = 0.1 
total_size= 5000                  
batch_size= 5 
img_size = 100
augment_factor = 1
train_infos = {}        
nLayer = 8
lam = 0.05
max_angle = 720
K = 8
epochs = 50
lr = 5e-4

shrink = 0.5
test_loss_dict = {} 

train_name_modl = 'Optimization_Projections_MODL_Test62'
train_name_unet = 'Optimization_Projections_UNet_Test62'

for projection in proj_num:

    tensor_path = datasets_folder+'Proj_{}_augmentFactor_{}_totalSize_{}_'.format(projection, augment_factor, total_size)

    datasets = modutils.formRegDatasets(folder_paths, img_resize =img_size)
    dataloaders = modutils.formDataloaders(datasets, proj_num, total_size, train_factor, val_factor, test_factor, batch_size, img_size, tensor_path, augment_factor, load_tensor = True, save_tensor = False)    

    # Training MODL
    model_MODL = modl.OPTmodl(nLayer, K, max_angle, projection, img_size, None, lam, results_folder, shared = True, unet_options = False)

    loss_fn = torch.nn.MSELoss(reduction = 'sum')
    loss_fbp_fn = torch.nn.MSELoss(reduction = 'sum')
    loss_backproj_fn = torch.nn.MSELoss(reduction = 'sum')
        
    optimizer = torch.optim.Adam(model_MODL.parameters(), lr = lr)
        
    model_MODL, train_info = modutils.model_training(model_MODL, loss_fn, loss_backproj_fn, loss_fbp_fn, optimizer, dataloaders, device, results_folder+train_name_modl, num_epochs = epochs, disp = True, do_checkpoint = 0, title = train_name_modl, plot_title = True, compute_mse = False, monai = False)

    train_infos[projection] = train_info
    
    with open(results_folder+train_name_modl+'LossProjections_Proj{}_nlay{}_epochs{}_K{}_lam{}_trnSize{}.pkl'.format(projection, nLayer, epochs, K, lam, train_factor), 'wb') as f:
    
        pickle.dump(train_infos, f)
        print('Diccionario salvado para proyección {}'.format(projection))
    
    modutils.save_net(model_folder+train_name_modl+'K_{}_lam_{}_nlay_{}_proj_{}'.format(K, lam, nLayer, projection), model_MODL)

    # Training UNet
    model_Unet = modl.UNet(1,1, residual = True, up_conv = True, batch_norm = True, batch_norm_inconv = True).to(device)
    
    loss_fn = torch.nn.L1Loss(reduction = 'mean')
    loss_fbp_fn = torch.nn.L1Loss(reduction = 'mean')
    loss_backproj_fn = torch.nn.L1Loss(reduction = 'mean')
    loss_mse = torch.nn.MSELoss(reduction = 'sum')
    
    optimizer_Unet = torch.optim.Adam(model_Unet.parameters(), lr = lr)
    
    model_Unet, train_info_Unet = modutils.model_training_unet(model_Unet, loss_fn, loss_fbp_fn, optimizer_Unet, dataloaders,  device, results_folder+train_name_unet, num_epochs = epochs, disp = True, monai = False)

    with open(results_folder+train_name_unet+'Train_UNet_lr{}_shrink{}.pkl'.format(lr, shrink), 'wb') as f:
    
        pickle.dump(train_info_Unet, f)
        print('Diccionario salvado para proyección {}'.format(projection))

    modutils.save_net(model_folder+train_name_unet+'Model_Unet_lr{}_shrink{}'.format(lr, shrink), model_Unet)

    test_loss_modl = []
    test_loss_fbp = []           
    test_loss_unet = []

    for inp, target, filt in tqdm(zip(dataloaders['test']['x'], dataloaders['test']['y'], dataloaders['test']['filtX'])): 
        
        pred_unet = model_Unet(inp)
        pred_modl = model_MODL(inp)

        loss_modl = loss_mse(pred_modl['dc'+str(K)], target)
        loss_fbp = loss_mse(filt, target)
        loss_unet = loss_mse(pred_unet, target)
                                                                 
        test_loss_modl.append(modutils.psnr(img_size, loss_modl.item(), 1))
        test_loss_fbp.append(modutils.psnr(img_size, loss_fbp.item(), 1))
        test_loss_unet.append(modutils.psnr(img_size, loss_unet.item(), 1))

    modutils.plot_outputs(target, pred_modl, results_folder+train_name_modl+'Test_images_proj{}_K{}.pdf'.format(projection, K))

    test_loss_dict[projection] = {'loss_modl': test_loss_modl, 'loss_fbp': test_loss_fbp, 'loss_unet':test_loss_unet}

    with open(results_folder+train_name_modl+'Projections_Proj{}_nLay{}_K{}_lam{}_trnSize{}.pkl'.format(projection, nLayer, K, lam, train_factor), 'wb') as f:
        
        pickle.dump(test_loss_dict, f)
        print('Diccionario salvado para proyección {}'.format(projection))
