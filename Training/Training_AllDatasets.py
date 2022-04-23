"""
Train the model with eight datasets and reserve three for testing
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
# Folder_paths_0 contains the datasets to be used for training, validation and test
folder_paths_0 = [f140114_5dpf, f140315_3dpf, f140419_5dpf, f140115_1dpf,f140714_5dpf]#, f140117_3dpf, f140114_5dpf] # Folders to be used

# Datasets to be used for testing
folder_paths_test = [f140117_3dpf]

# Train name
train_name = 'Test42_MODLNetwork'

#torch.autograd.detect_anomaly()
umbral_reg = 200

#%% Datasets 
# Training with more than one dataset
proj_num = 72

train_factor = 0.7
val_factor = 0.2
test_factor = 0.1
total_size = 5000
total_size_test = 2000

batch_size = 10
img_size = 100
augment_factor = 1
train_infos = {}
tensor_path = datasets_folder+'Proj_{}_augmentFactor_{}_totalSize_{}_'.format(proj_num, augment_factor, total_size)

#datasets = modutils.formRegDatasets(folder_paths, umbral_reg, img_resize = img_size)

dataloaders_0 = modutils.formDataloaders(folder_paths_0, proj_num, total_size, train_factor, val_factor, test_factor, batch_size, img_size, tensor_path, filter_dataset = '140117_3dpf', augment_factor=1, load_tensor = True, save_tensor = False)

dataloaders_test = modutils.formDataloaders(folder_paths_test, proj_num, total_size_test, 0.0, 0.0, test_factor, batch_size, img_size, tensor_path+'_testing', filter_dataset = '140117_3dpf', augment_factor = 1, load_tensor = True, save_tensor =False)

# Network parameters
lam = 0.05
nLayer = 8
K = 10
epochs = 30
max_angle = 640

# Model
model = modl.OPTmodl(nLayer, K, max_angle, proj_num, img_size, None, lam, True, results_folder, useUnet = False)

loss_fn = torch.nn.MSELoss(reduction = 'sum')
loss_fbp_fn = torch.nn.MSELoss(reduction = 'sum')
loss_backproj_fn = torch.nn.MSELoss(reduction = 'sum')
lr = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr = lr)

model, train_info = modutils.model_training(model, loss_fn, loss_backproj_fn, loss_fbp_fn, optimizer, dataloaders_0, device, results_folder+train_name, num_epochs = epochs, disp = True, do_checkpoint = 0, title = train_name, plot_title = True )

with open(results_folder+train_name+'Dict_Proj{}_nlay{}_epochs{}_K{}_lam{}_trnSize{}.pkl'.format(proj_num, nLayer, epochs, K, lam, train_factor), 'wb') as f:
 
     pickle.dump(train_info, f)
     print('Diccionario salvado para proyección {}'.format(proj_num))

modutils.save_net(model_folder+'Lambdas_K_{}_lam_{}_nlay_{}_proj_{}_trnSize{}'.format(K, lam, nLayer, proj_num, train_factor), model)

modutils.load_net(model_folder+'Lambdas_K_{}_lam_{}_nlay_{}_proj_{}_trnSize{}'.format(K, lam, nLayer, proj_num, train_factor), model, device)
### Testing part
test_loss_total = []
test_loss_fbp_total = []
test_loss_dict = {}

for inp, target, filtX in tqdm(zip(dataloaders_test['test']['x'], dataloaders_test['test']['y'], dataloaders_test['test']['filtX'])): 
    
    pred = model(inp)
    loss_test = loss_fn(pred['dc'+str(K)], target)
    loss_test_fbp = loss_fbp_fn(inp, target)
    
    test_loss_total.append(modutils.psnr(img_size, loss_test.item(), 1))
    test_loss_fbp_total.append(modutils.psnr(img_size, loss_test_fbp.item(), 1))

modutils.plot_outputs(target, pred, results_folder+train_name+'Test_images_proj{}.pdf'.format(proj_num))                                                                                                                      
test_loss_dict[proj_num] = {'loss_net': test_loss_total, 'loss_fbp': test_loss_fbp_total}
                                                                                                                                                         
with open(results_folder+train_name+'Proj{}_nLay{}_epochs{}_K{}_lam{}_trnSize{}.pkl'.format(proj_num, nLayer, epochs, K, lam, train_factor), 'wb') as f:
    
    pickle.dump(test_loss_dict, f)
    print('Diccionario salvado para proyección {}'.format(proj_num))
