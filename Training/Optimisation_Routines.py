"""
Optimise parameters using 
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
from bayes_opt import BayesianOptimization

# Using CPU or GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
folder_paths = [ f140315_3dpf, f140419_5dpf, f140115_1dpf,f140714_5dpf] # Folders to be used

umbral_reg = 50

#%% Datasets 
# Training with more than one dataset
proj_num = 72

train_size = 100
val_size = 20
test_size = 20

batch_size = 5
img_size = 100
augment_factor = 15
train_infos = {}
test_loss_dict = {}

train_dataset, test_dataset = modutils.formRegDatasets(folder_paths, umbral_reg, img_resize = img_size)
train_name = 'OptimizationLambda_Test14'

# Load desired and undersampled datasets, on image space. Testing on Test Dataset
dataloaders = modutils.formDataloaders(train_dataset, test_dataset, proj_num, train_size, val_size, test_size, batch_size, img_size, augment_factor)

print('Lambda Optimization begins...')
#  Optimization function for lambda
def lambda_optimization(lambdas):
    
    #%% Model Settings
    nLayer= 4
    K = 10
    proj_num = 72
    epochs = 20
    max_angle = 640
    img_size = 100
    lr = 1e-3
    
    model = modl.OPTmodl(nLayer, K, max_angle, proj_num, img_size, None, lambdas, True, results_folder)
    loss_fn = torch.nn.MSELoss(reduction = 'sum')
    loss_fbp_fn = torch.nn.MSELoss(reduction = 'sum')
    loss_backproj_fn = torch.nn.MSELoss(reduction = 'sum')
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
        
    #### Training
    model, train_info = modutils.model_training(model, loss_fn, loss_backproj_fn, loss_fbp_fn, optimizer, dataloaders, device, results_folder+train_name, num_epochs = epochs, disp = True, do_checkpoint = 0, plot_title = False, title = train_name)
    
    return -train_info['val'][-1]

lambda_bounds = {'lambdas':(0.01, 20)}
optimizer = BayesianOptimization(
        f = lambda_optimization,
        pbounds = lambda_bounds,
        random_state = 1, 
        verbose = 2)

init_points = 5
n_iter = 5

optimizer.maximize(init_points = init_points, n_iter = n_iter)

with open(results_folder+'OptimizationValues_'+train_name+'.pkl', 'wb') as f:

    pickle.dump(optimizer.res, f)


