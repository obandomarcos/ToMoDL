"""
Train the model initialising weights with a previous network
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

umbral_reg = 50

#%% Datasets 
# Training with more than one dataset

number_projections = 72
total_size = 2000
train_factor = 0.7
val_factor = 0.2
test_factor = 0.1
batch_size = 5
img_size = 100
projections_augment_factor = 3
transform_augment_factor = 3

tensor_path = datasets_folder+'Proj_{}_size_{}_imgsize_{}_projaugment_{}_transformaugment_{}_'.format(number_projections, total_size, img_size, projections_augment_factor, transform_augment_factor)

#datasets = modutils.formRegDatasets(folder_paths, umbral_reg, img_resize = img_size)
datasets = []

dataloaders = modutils.formDataloaders(datasets, number_projections, total_size, projections_augment_factor, transform_augment_factor, train_factor, val_factor, test_factor, img_size, batch_size, tensor_path = tensor_path, load_tensor = True, save_tensor = False)

train_infos = {}
test_loss_dict = {}

lambdas = [0.05]
train_name = 'NBN_DataAugmentation_Test18'

for lam in lambdas:
    
    # Load desired and undersampled datasets, on image space. Testing on Test Dataset
    #%% Model Settings
    nLayer= 4
    K = 10
    epochs = 20
    max_angle = 640
    
    model = modl.OPTmodl(nLayer, K, max_angle, number_projections, img_size, None, lam, True, results_folder)
    loss_fn = torch.nn.MSELoss(reduction = 'sum')
    loss_fbp_fn = torch.nn.MSELoss(reduction = 'sum')
    loss_backproj_fn = torch.nn.MSELoss(reduction = 'sum')
    lr = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
        
    #### Training
    model, train_info = modutils.model_training(model, loss_fn, loss_backproj_fn, loss_fbp_fn, optimizer, dataloaders, device, results_folder+train_name, num_epochs = epochs, disp = True, do_checkpoint = 0, title = train_name, plot_title = True )
    # 
    train_infos[K] = train_info
    #
    print('Train MODL loss {}'.format(train_infos[K]['train'][-1]))
    print('Train FBP loss {}'.format(train_infos[K]['train_fbp'][-1]))

    #print('Second training, K = 10')
    #model.K = 10
    #K = 10
    #epochs = 20
    #model, train_info = modutils.model_training(model, loss_fn, loss_backproj_fn, loss_fbp_fn, optimizer, dataloaders, device, results_folder, num_epochs = epochs, disp = True, do_checkpoint = 0, title = train_name, plot_title = True)
    
    #train_infos[K] = train_info

    ##%% save loss for fbp and modl network
    with open(results_folder+train_name+'Dict_Proj{}_nlay{}_epochs{}_K{}_lam{}_trnSize{}.pkl'.format(number_projections, nLayer, epochs, K, lam, train_factor), 'wb') as f:
    #
        pickle.dump(train_infos, f)
        print('Diccionario salvado para proyección {}'.format(number_projections))
    #
    modutils.save_net(model_folder+'Lambdas_K_{}_lam_{}_nlay_{}_proj_{}_trnSize{}'.format(K, lam, nLayer, number_projections, train_factor), model)

    ### Testing part
    test_loss_total = []
    test_loss_fbp_total = []

    for inp, target in tqdm(zip(dataloaders['test']['x'], dataloaders['test']['y'])): 
        
        pred = model(inp)
        loss_test = loss_fn(pred['dc'+str(K)], target)
        loss_test_fbp = loss_fbp_fn(inp, target)
        
        test_loss_total.append(modutils.psnr(img_size, loss_test.item(), 1))
        test_loss_fbp_total.append(modutils.psnr(img_size, loss_test_fbp.item(), 1))
    
    modutils.plot_outputs(target, pred, results_folder+train_name+'Test_images_proj{}.pdf'.format(number_projections))

    test_loss_dict[number_projections] = {'loss_net': test_loss_total, 'loss_fbp': test_loss_fbp_total}

    with open(results_folder+train_name+'Proj{}_nLay{}_epochs{}_K{}_lam{}_trnSize{}.pkl'.format(number_projections, nLayer, epochs, K, lam, train_factor), 'wb') as f:
        
        pickle.dump(test_loss_dict, f)
        print('Diccionario salvado para proyección {}'.format(number_projections))

    del model
