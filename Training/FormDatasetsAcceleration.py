"""
Train the model with different K
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
folder_paths = [f140115_1dpf, f140315_3dpf, f140419_5dpf, f140714_5dpf, f140117_3dpf, f140114_5dpf] # Folders to be used

total_num = 720
proj_numbers = 720//np.arange(22, 26, 2)
factors = np.arange(22, 26, 2)
total_size=5000                  
img_size = 100
batch_size = 5
train_factor = 0.7
val_factor = 0.2
test_factor = 0.1 
augment_factor = 1

for acc_factor, proj_num in zip(factors, proj_numbers):
    
    tensor_path = datasets_folder+'Proj_{}_augmentFactor_{}_totalSize_{}_'.format(proj_num, augment_factor, total_size)                                                                                                                                     
    datasets = modutils.formRegDatasets(folder_paths, img_resize =img_size)
    dataloaders = modutils.formDataloaders(datasets, proj_num, total_size, train_factor, val_factor, test_factor, batch_size, img_size, tensor_path, augment_factor, load_tensor = False, save_tensor = True)    

    print(tensor_path+' saved!')
    del datasets, dataloaders

