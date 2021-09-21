"""
Train the model with same image with random subsampling
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
train_dataset, val_dataset = modutils.formRegDatasets(folder_paths, umbral_reg)     # Formo los datasets registrados

#%% Datasets 
# Training with more than one dataset
number_projections = [20, 30, 40, 60, 90, 120, 180]
total_size = 1000
train_size = int(total_size*0.8)
val_size = int(total_size*0.2)
test_size = int(total_size*0.1)
batch_size = 5

img_resize = 100
slice_idx = 200

slice_test = 100
test_loss_dict = {}
test_loss_dict['img_size'] = img_resize

for proj_num in number_projections:

    train_unique_dataset, train_target, maxAngle = modutils.formUniqueDataset(train_dataset, total_size, proj_num, slice_idx, img_resize)
    
    ### TRAIN
    #%% Model Settings
    nLayer = 4
    K = 2
    epochs = 50
    lam = 0.05
    
    model = modl.OPTmodl(nLayer, K, maxAngle, img_resize, None, lam)
    loss_fn = torch.nn.MSELoss(reduction = 'sum')
    loss_fbp_fn = torch.nn.MSELoss(reduction = 'sum') 
    lr = 1e-3
    optimizer = torch.optim.RMSprop(model.parameters())
    
    trainX = torch.utils.data.DataLoader(train_unique_dataset[:train_size,:,:,:], batch_size = batch_size, shuffle = False, num_workers = 0)
    valX = torch.utils.data.DataLoader(train_unique_dataset[train_size:train_size+val_size,:,:,:], batch_size = batch_size, shuffle = False, num_workers = 0)

    dataloader = {'train':trainX, 'val':valX}
    
    # Train network with same data
    model, train_info = modutils.unique_model_training(model, loss_fn, loss_fbp_fn, optimizer, dataloader, train_target, epochs, device, batch_size, disp = True)

    with open(results_folder+'trainInfo_Proj{}_K{}_lam{}.pkl'.format(proj_num, K, lam), 'wb') as f:
        
        pickle.dump(train_info, f)
        print('Diccionario de entrenamiento salvado para proyección {}'.format(proj_num))
 
    del dataloader

    ### TEST
    # Agarro un test dataset con otra imagen
    # Random slices
    random_slices = np.random.choice(range(train_unique_dataset.shape[2]), test_size, replace = False)
    
    test_loss_total = []
    test_loss_fbp_total = []

    # por cada slice, formo un dataset y reconstruyo con el denoiser aprendido
    for slice_test in tqdm(random_slices):

        test_dataset, test_target, _ = modutils.formUniqueDataset(train_dataset, 1, proj_num, slice_test, img_resize)
        test_dataset = torch.utils.data.DataLoader(test_dataset, batch_size = 1, shuffle = False, num_workers = 0)
        test_target = torch.unsqueeze(torch.unsqueeze(test_target, 0), 0)
        
        loss_test_sum = 0.0
        loss_test_fbp_sum = 0.0

        # paso los inputs y luego comparo con el objetivo SOLO HAY UN INPUT
        for inputs in test_dataset:
            
            pred = model(inputs)
            loss_test = loss_fn(pred['dc'+str(K-1)], test_target)
            loss_test_fbp = loss_fbp_fn(inputs, test_target)
            
            test_loss_total.append(loss_test.item()) 
            test_loss_fbp_total.append(loss_test_fbp.item())

    # loss de los ejemplos de testeo. 
    
    test_loss_dict[proj_num] = {'loss_net': test_loss_total, 'loss_fbp':test_loss_fbp_total}
        
    modutils.plot_outputs(test_target,pred, results_folder+'Test_Unique_slice{}_proj_{}.pdf'.format(slice_test, proj_num))

    with open(results_folder+'Unique_Proj{}_K{}_lam{}.pkl'.format(proj_num, K, lam), 'wb') as f:
        
        pickle.dump(test_loss_dict, f)
        print('Diccionario salvado para proyección {}'.format(proj_num))
        #print(test_loss_dict)
        
    modutils.save_net(model_folder+'Unique_K_{}_lam_{}_nlay_{}_proj_{}'.format(K, lam, nLayer, proj_num), model)

    modutils.plot_histogram(test_loss_dict[proj_num], test_loss_dict['img_size'], results_folder+'Histogram_proj{}.pdf'.format(proj_num))
