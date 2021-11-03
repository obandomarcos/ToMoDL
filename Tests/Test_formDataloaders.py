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
total_size = 3000
train_factor = 0.7
val_size = 0.2
test_size = 0.1

batch_size = 5
img_size = 100
augment_factor = 15
train_infos = {}
test_loss_dict = {}

datasets = modutils.formRegDatasets(folder_paths, umbral_reg, img_resize = img_size)

dataloaders = modutils.formDataloaders(datasets, number_projections, total_size, projections_augment_factor, train_factor, val_factor, test_factor, img_size, batch_size, load_tensor = False, save_tensor = True)

