"""
Test how to register all the datasets I have
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#%% Memory usage

def memory_limit():
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (get_memory() * 1024 *7/ 8, hard))

def get_memory():
    with open('/proc/meminfo', 'r') as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                free_memory += int(sline[1])
    return free_memory

#%% 
#memory_limit()    
#%%
folder_paths = [f140117_3dpf, f140114_5dpf, f140315_3dpf]

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

#datasets = modutils.formRegDatasets(folder_paths, umbral_reg, img_resize = img_size, sample = 'head')
#dataloaders = modutils.formDataloaders(folder_paths, proj_num, total_size, train_factor, val_factor, test_factor, batch_size, img_size, tensor_path, augment_factor, load_tensor = False, save_tensor = True)
datasets = [modutils.openDataset(f) for f in folder_paths]

fig, ax = plt.subplots(3,1)

ax[0].set_title('Projection 180')
ax[0].imshow(np.flipud(datasets[0][datasets[0].shape[0]//2,:,:]))
ax[1].set_title('Projection 0')
ax[1].imshow(datasets[0][0,...])
ax[2].set_title('Difference')
ax[2].imshow(datasets[0][0,:,:]-np.flipud(datasets[0][datasets[0].shape[0]//2,:,:]))

fig.savefig(results_folder+folder_paths[0][-11:]+'_RegistrationSuperposed_Test37.pdf', bbox_inches = 'tight')

# Reconstruction

n_angles = 640
det_count = int(img_size*np.sqrt(2))
angles = np.linspace(0, 2*180, n_angles, endpoint = False)

torch.cuda.empty_cache()
radon = Radon(img_size, angles, clip_to_circle = False, det_count = det_count)
sinos = [dataset[:640,:,200].T for dataset in datasets]

recs = [iradon(sino, circle = False, theta = angles ) for sino in sinos]

fig, ax = plt.subplots(1,3)

for rec, a, title in zip(recs, ax, folder_paths):
    print(rec.shape) 
    a.set_title(title[-11:])
    a.imshow(rec)

fig.savefig(results_folder+'Reconstruction_Registration_Test38.pdf', bbox_inches = 'tight')


sinos_full = [datasets[0][640*i:640*(i+1), :, 200].T for i in range(4)]

for sino in sinos_full: print(sino.shape)

recs_full = [iradon(sino, circle = False, theta = angles) for sino in sinos_full]

fig, ax = plt.subplots(len(sinos_full), 1)

parts = ['head', 'body', 'tail', 'tail 2']

for a, part, rec in zip(ax, parts, recs_full):

    a.imshow(rec)

fig.savefig(results_folder+'Reconstruccion_Registration_FishParts_Test39.pdf', bbox_inches = 'tight')





