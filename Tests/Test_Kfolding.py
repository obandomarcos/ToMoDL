'''
Test K-folding over volumes train dataset
    * Writes images from each
author: obanmarcos
'''

import os
import os,time, sys
os.chdir('/home/obanmarcos/Balseiro/DeepOPT/')
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
import cv2

# Using CPU or GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#%% Datasets 
# Training with more than one dataset
# Projnum == %10 of the data

train_factor = 0.7
val_factor = 0.2
test_factor = 0.1 
total_size= 5000                 
batch_size= 5 
img_size = 100
augment_factor = 1
train_infos = {}        
projection = 72

train_name_modl = 'KFold_PSNR_MODL_Test70'
train_name_SSIM = 'KFold_SSIM_MODL_Test70'

test_models = False                 

def k_folding_test(projection, augment_factor, total_size, train_factor, val_factor, test_factor, batch_size, img_size, datasets, k_iters, k_fold_datasets = 2):
    '''
    Iterates k-folding order 
    - k_iters is the number of folds to be done to the dataset
    - k_fold_datasets is the number of datasets to be taken for k-folding
    '''
    for _ in range(k_iters):

        datasets = modutils.k_fold_list(datasets, k_fold_datasets)
        tensor_path = None

        dataloaders = dlutils.formDataloaders(datasets, projection, total_size, train_factor, val_factor, test_factor, batch_size, img_size, tensor_path, augment_factor, load_tensor = False, save_tensor = False, k_fold_datasets = k_fold_datasets)
        
        # Images from testing should be from the penultimate dataset of the list (considering k_fold_datasets == 2)
        test_image = (255.0*next(iter(dataloaders['test']['y'])).cpu().detach().numpy()[0, 0,...]).astype(int)
        
        test_image_path = datasets[-2].split('/')[-1].replace('.pkl', '')
        print(test_image_path)

        cv2.imwrite('./Tests/ResultadosKFold/'+test_image_path+'.jpg', test_image)

if __name__ == '__main__':

    folder_paths = [f140115_1dpf, f140315_3dpf, f140419_5dpf, f140714_5dpf, f140117_3dpf, f140114_5dpf]
    k_iters = 6
    
    datasets = dlutils.formRegDatasets(folder_paths, fish_parts = None)

    k_folding_test(projection, augment_factor, total_size, train_factor, val_factor, test_factor, batch_size, img_size, datasets, k_iters, k_fold_datasets = 2)