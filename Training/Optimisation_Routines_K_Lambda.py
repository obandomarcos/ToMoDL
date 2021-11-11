"""
Optimise parameters K and Lambda simultaneousy 
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
proj_num = 72

train_factor = 0.7
val_factor = 0.2
test_factor = 0.1
total_size = 2000

batch_size = 5
img_size = 100
augment_factor = 15
train_infos = {}
test_loss_dict = {}
tensor_path = datasets_folder+'Proj_{}_augmentFactor_{}_totalSize_{}_'.format(proj_num, augment_factor, total_size)

datasets = []
#datasets = modutils.formRegDatasets(folder_paths, umbral_reg, img_resize = img_size)

dataloaders = modutils.formDataloaders(datasets, proj_num, total_size, train_factor, val_factor, test_factor, batch_size, img_size, tensor_path, augment_factor, load_tensor = True, save_tensor = False)

train_name = 'OptimizationLambdaK_Test24'

def lambda_K_optimization(lambdas, K):
    
    #%% Model Settings
    nLayer= 4
    K = int(K)
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

parameters_bounds = {'lambdas':(0.01, 2.5), 'K' : (1,12)}

optimizer = BayesianOptimization(
        f = lambda_K_optimization,
        pbounds = parameters_bounds,
        random_state = 43, 
        verbose = 2)

init_points = 10
n_iter = 20

optimizer.maximize(init_points = init_points, n_iter = n_iter, acq = "ucb", kappa = 10)

with open(results_folder+'OptimizationValues_'+train_name+'.pkl', 'wb') as f:

    pickle.dump(optimizer.res, f)
