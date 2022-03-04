"""
Train the model with different projections angles
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
folder_paths = [ f140315_3dpf, f140419_5dpf, f140115_1dpf,f140714_5dpf] # Folders to be used

# Projnum == %10 of the data
proj_num = 72
Ks = np.arange(1,11)

train_factor = 0.7
val_factor = 0.2
test_factor = 0.1 
total_size= 5000                  
batch_size= 5 
img_size = 100
augment_factor = 1
K = 10

lr = 5e-4
lam = 0.05
max_angle = 640
nLayer = 8
epochs = 3
shrink = 0.5

train_infos = {}        
test_loss_dict_ModlUNet = {}
test_loss_dict_UNet = {}

tensor_path = datasets_folder+'Proj_{}_augmentFactor_{}_totalSize_{}_'.format(proj_num, augment_factor, total_size)                                                                                                      
datasets = modutils.formRegDatasets(folder_paths, img_resize =img_size)
dataloaders = modutils.formDataloaders(datasets, proj_num, total_size, train_factor, val_factor, test_factor, batch_size, img_size, tensor_path, augment_factor, load_tensor = True, save_tensor = False)    

train_name_ModlUnet = 'UnetModl_CorrectRegistration_Test53'
train_name_Unet = 'Unet_CorrectRegistration_Test54'

#  Train model + UNET plugged
model_ModlUnet = modl.OPTmodl(nLayer, K, max_angle, proj_num, img_size, None, lam, True, results_folder, useUnet = True)
model_Unet = modl.UNet(1,1)

loss_fn = torch.nn.MSELoss(reduction = 'sum')
loss_fbp_fn = torch.nn.MSELoss(reduction = 'sum')
loss_backproj_fn = torch.nn.MSELoss(reduction = 'sum')
optimizer_ModlUnet = torch.optim.Adam(model_ModlUnet.parameters(), lr = lr)
optimizer_Unet = torch.optim.Adam(model_Unet.parameters(), lr = lr)

# model_ModlUnet, train_info_ModlUnet = modutils.model_training(model_ModlUnet, loss_fn, loss_backproj_fn, loss_fbp_fn, optimizer_ModlUnet, dataloaders, device, results_folder+train_name_ModlUnet, num_epochs = epochs, disp = True, do_checkpoint = 0, title = train_name_ModlUnet, plot_title = True)

# print('Train MODL+UNet loss {}'.format(train_info_ModlUnet['train'][-1]))
# print('Train FBP loss {}'.format(train_info_ModlUnet['train_fbp'][-1]))
# #%% save loss for fbp and modl network
# with open(results_folder+train_name_ModlUnet+'ModlUNet_lr{}_shrink{}.pkl'.format(lr, shrink), 'wb') as f:

#     pickle.dump(train_info_ModlUnet, f)
#     print('Diccionario salvado para proyección {}'.format(proj_num))

# modutils.save_net(model_folder+train_name_ModlUnet+'_MoDLUNet_lr{}_shrink{}'.format(lr, shrink), model_ModlUnet)

#  Train directly with Unet (inputs change)
model_Unet, train_info_Unet = modutils.model_training_unet(model_Unet, loss_fn, loss_fbp_fn, optimizer_Unet, dataloaders,  device, results_folder+train_name_Unet, num_epochs = epochs, disp = True)

print('Train UNET loss {}'.format(train_info_Unet['train'][-1]))
print('Train FBP loss {}'.format(train_info_Unet['train_fbp'][-1]))

with open(results_folder+train_name_Unet+'UNet_lr{}_shrink{}.pkl'.format(lr, shrink), 'wb') as f:

    pickle.dump(train_info_Unet, f)
    print('Diccionario salvado para proyección {}'.format(proj_num))

### Testing part
test_loss_total_ModlUnet = []
test_loss_total_Unet = []
test_loss_fbp_total = []           
test_loss_backproj_total = []

for inp, target, filt in tqdm(zip(dataloaders['test']['x'], dataloaders['test']['y'], dataloaders['test']['filtX'])): 
    
    pred_ModlUnet = model_ModlUnet(inp)
    pred_Unet = model_Unet(filtX) # Input is FBP with less projections

    loss_test_ModlUnet = loss_fn(pred_ModlUnet['dc'+str(K)], target).item()
    loss_test_Unet = loss_fn(pred_Unet, target).item()
    
    loss_test_backproj = loss_backproj_fn(inp, target)
    loss_test_fbp = loss_fbp_fn(filt, target)                                                                                            
    test_loss_total_ModlUnet.append(modutils.psnr(img_size, loss_test_ModlUnet, 1))
    test_loss_total_Unet.append(modutils.psnr(img_size, loss_test_Unet, 1))
    test_loss_fbp_total.append(modutils.psnr(img_size, loss_test_fbp.item(), 1))
    test_loss_backproj_total.append(modutils.psnr(img_size, loss_test_backproj.item(), 1))

modutils.plot_outputs(target, pred_ModlUnet, results_folder+train_name+'Test_images_MODL_UNet_lr{}_shrink{}.pdf'.format(lr, shrink))
modutils.plot_outputs(target, pred_Unet, results_folder+train_name+'Test_images_UNet_lr{}_shrink{}.pdf'.format(lr, shrink))

test_loss_dict_ModlUNet = {'loss_net': test_loss_total_ModlUnet, 'loss_fbp': test_loss_fbp_total, 'loss_backproj':test_loss_backproj_total}
test_loss_dict_UNet = {'loss_net': test_loss_total_Unet, 'loss_fbp': test_loss_fbp_total, 'loss_backproj':test_loss_backproj_total}
                                                            
with open(results_folder+train_name_ModlUnet+'_UnetMoDL_lr{}_shrink{}.pkl'.format(lr, shrink), 'wb') as f:
    
    pickle.dump(test_loss_dict_ModlUNet, f)
    print('Diccionario salvado para proyección {}, MODL+UNET'.format(proj_num))

with open(results_folder+train_name_Unet+'_Unet_lr{}_shrink{}.pkl'.format(lr, shrink), 'wb') as f:
    
    pickle.dump(test_loss_dict_UNet, f)
    print('Diccionario salvado para proyección {}, UNET'.format(proj_num))