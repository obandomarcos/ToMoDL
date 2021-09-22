"""
Load model
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

umbral_reg = 50
train_dataset, test_dataset = modutils.formRegDatasets(folder_paths, umbral_reg)

proj_num = 72
train_size = 1000            
val_size = 200
test_size = 200

batch_size = 5
img_size = 100

dataloaders = modutils.formDataloaders(train_dataset, test_dataset, proj_num, train_size, val_size, test_size, batch_size, img_size)

#%% Model Settings
nLayer = 3
K = 5
epochs = 40
lam = 0.05
max_angle = 720

model = modl.OPTmodl(nLayer, K, max_angle, img_size, None, lam)

modutils.load_net(model_folder+'K_{}_lam_{}_nlay_{}_proj_{}'.format(K, lam, nLayer, proj_num), model, device)

train_input = next(iter(dataloaders['train']['x']))
train_target = next(iter(dataloaders['train']['y']))
train_pred = model(train_input)

#test_input = next(iter(dataloaders['val']['x']))
#test_target = next(iter(dataloaders['val']['y']))
#test_pred = model(test_input)

modutils.plot_data(train_pred['dc0'], train_target, results_folder+'Train_comparison.pdf')
modutils.plot_outputs(train_target, train_pred, results_folder+'Train_images_epoch{}_proj{}.pdf'.format(epochs, model.nAngles))

#modutils.plot_outputs(test_target, test_pred, results_folder+'Test_images_epoch{}_proj{}.pdf'.format(epochs, model.nAngles))
