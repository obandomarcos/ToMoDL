"""
Test interpolation of smaller datasets
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
folder_paths = [f140114_5dpf]# f140315_3dpf, f140419_5dpf, f140115_1dpf,f140714_5dpf] # Folders to be used

umbral_reg = 200

#%% Datasets 
# Training with more than one dataset
proj_num = 72

train_factor = 0.7
val_factor = 0.2
test_factor = 0.1
total_size = 3000

batch_size = 5
img_size = 100
augment_factor = 2
train_infos = {}
test_loss_dict = {}
tensor_path = datasets_folder+'Proj_{}_augmentFactor_{}_totalSize_{}_'.format(proj_num, augment_factor, total_size)

#datasets = modutils.formRegDatasets(folder_paths, umbral_reg, img_resize = img_size)
datasets = []

dataloaders = modutils.formDataloaders(datasets, proj_num, total_size, train_factor, val_factor, test_factor, batch_size, img_size, tensor_path, augment_factor, load_tensor = True, save_tensor = False)

train_name = 'TestingBatch_Test36'

fig, axs = plt.subplots(3,3)

for a, img in zip(axs.flatten(), dataloaders['train']['x']):
        
    img = img[0,0,...].detach().cpu().numpy()
    a.imshow(img)

fig.savefig(results_folder+train_name+'Image.pdf')

#datasets = []

#dataloaders = modutils.formDataloaders(datasets, proj_num, total_size, train_factor, val_factor, test_factor, batch_size, img_size, tensor_path, augment_factor, load_tensor = True, save_tensor = False)
