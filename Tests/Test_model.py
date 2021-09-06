'''
Test: Model training 
author: obanmarcos 
'''
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

# Using CPU or GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
folder_paths = [f140115_1dpf, f140315_3dpf, f140419_5dpf, f140714_5dpf] # Folders to be used

umbral_reg = 50
train_dataset, test_dataset = modutils.formRegDatasets(folder_paths, umbral_reg)
print(test_dataset[0].shape)
#%% Datasets 
# Training with more than one dataset
number_projections = 90
train_size = 200
test_size = 200
batch_size = 5
img_size = 200

dataloaders = modutils.formDataloaders(train_dataset, test_dataset, number_projections, train_size, test_size, batch_size, img_size)

#%% Model Settings
nLayer = 4
K = 10
epochs = 10
lam = 0.01
maxAngle = 360

model = modl.OPTmodl(nLayer, K, maxAngle, number_projections, img_size, None, lam)
loss_fn = torch.nn.MSELoss(reduction = 'sum')
lr = 1e-3
optimizer = torch.optim.RMSprop(model.parameters())

model, train_info = modutils.model_training(model, loss_fn, optimizer, dataloaders, device, model_folder, num_epochs = 25, disp = True, do_checkpoint = 4)


