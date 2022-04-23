"""
Test Registration errors for each dataset I have
"""
#%%
import os
import os,time, sys
os.chdir('/home/marcos/DeepOPT/')
sys.path.append('./Utilities/')
sys.path.append('./OPTmodl/')
import ModelUtilities as modutils
import numpy as np
import time
import copy 
import datetime
import sys, os
from torch_radon import Radon, RadonFanbeam
import DataLoading as DL
import math
import matplotlib.pyplot as plt
import cv2 
from Folders_cluster import *
import resource
import torch
from skimage.transform import radon, iradon
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

folder_paths = [f140117_3dpf, f140114_5dpf, f140315_3dpf, f140419_5dpf, f140115_1dpf,f140714_5dpf]

umbral_reg = 200

proj_num = 72

train_factor = 0.7
val_factor = 0.2
test_factor = 0.1
total_size = 10
batch_size = 5
img_size = 100
augment_factor = 2
train_infos = {}
test_loss_dict = {}
tensor_path = datasets_folder+'Proj_{}_augmentFactor_{}_totalSize_{}_'.format(proj_num, augment_factor, total_size)
load_tensor = True
save_tensor = True

#datasets = modutils.formRegDatasets(folder_paths, umbral_reg, img_resize = img_size, sample = None)
#dataloaders = modutils.formDataloaders(folder_paths, proj_num, total_size, train_factor, val_factor, test_factor, batch_size, img_size, tensor_path, augment_factor, load_tensor = False, save_tensor = True)
datasets_raw = [modutils.openDataset(f) for f in folder_paths]
parts = ['head', 'body', 'tail', 'tail 2']

datasets = []

for dataset, fp in zip(datasets_raw, folder_paths):

    if isinstance(dataset, list):
        
        for dataset_part, part in zip(dataset, parts):

            datasets.append((dataset_part, fp[-11:]+' '+part))

    else:

        datasets.append((dataset, ' '+fp[-11:]))

def grabPairs(dataset, frames):

    l = dataset.shape[0]//2
    
    rand = random.sample(range(l), frames)
    psnr = []

    for r in rand:

        im1 = dataset[r, ...]
        im2 = np.flipud(dataset[r+l,...])
        mse = ((im1-im2)**2).sum()/im1.size
        psnr.append(10*np.log10((im1.max()-im2.min())**2/mse))
    
    return np.array(psnr).mean(), np.array(psnr).std()

#print(datasets[0])
datasets_mse = [(*grabPairs(dataset[0], 20), dataset[1]) for dataset in datasets]

print([dataset[0] for dataset in datasets_mse])
fig, ax = plt.subplots(1,1)

ax.plot([mse[0] for mse in datasets_mse])
ax.errorbar(x = np.arange(0, len(datasets_mse)), y = [mse[0] for mse in datasets_mse], yerr = [mse[1] for mse in datasets_mse],  capsize=3, capthick=3)
ax.set_xticks(np.arange(0, len(datasets_mse)))
ax.set_ylabel('PSNR [dB]')
ax.set_title('Registration error within datasets')
ax.set_xticklabels([mse[2] for mse in datasets_mse], rotation='vertical')

fig.savefig(results_folder+'Test40_RegistrationError.pdf', bbox_inches = 'tight')







