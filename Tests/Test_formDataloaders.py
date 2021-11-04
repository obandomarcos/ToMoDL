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

datasets = modutils.formRegDatasets(folder_paths, umbral_reg, img_resize = img_size)
#datasets = []

dataloaders = modutils.formDataloaders(datasets, number_projections, total_size, projections_augment_factor, transform_augment_factor, train_factor, val_factor, test_factor, img_size, batch_size, tensor_path = tensor_path, load_tensor = False, save_tensor = True)

j = 3

for phase in ['train', 'val', 'test']:

    fig, ax = plt.subplots(j,3)
    
    for i,(x, y, filtx) in enumerate(zip(dataloaders[phase]['x'], dataloaders[phase]['y'], dataloaders[phase]['filtX'])):
    
        if i == j:
            break
        
        ax[i,0].imshow(x.numpy()[0,0,:,:])
        ax[i,1].imshow(y.numpy()[0,0,:,:]) 
        ax[i,2].imshow(filtx.numpy()[0,0,:,:])
    
    ax[0,0].set_title('Input')
    ax[0,1].set_title('Target')
    ax[0,2].set_title('Filtered backprojection')
    
    fig.savefig(results_folder+'Test17_DataAugmentation_{}.pdf'.format(phase))


