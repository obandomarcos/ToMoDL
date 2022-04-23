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

datasets_raw = [modutils.openDataset(f) for f in folder_paths]
parts = ['head', 'body', 'tail', 'tail 2']

datasets = []

for dataset, fp in zip(datasets_raw, folder_paths):

    if isinstance(dataset, list):
        
        for dataset_part, part in zip(dataset, parts):

            datasets.append((dataset_part, fp[-11:]+' '+part))

    else:

        datasets.append((dataset, ' '+fp[-11:]))

n_angles = 640
det_count = int(img_size*np.sqrt(2))
angles = np.linspace(0, 2*180, n_angles, endpoint = False)

torch.cuda.empty_cache()
radon = Radon(img_size, angles, clip_to_circle = False, det_count = det_count)

fig, ax = plt.subplots(4,len(datasets)//4, constrained_layout = True)
fig2, ax2 = plt.subplots(4, len(datasets)//4, constrained_layout = True)

for a, a2, dataset in zip(ax.flatten(), ax2.flatten(), datasets):
    
    sino = dataset[0][:,:,200].T
    im1 = dataset[0][0,:,:]
    im2 = np.flipud(dataset[0][640//2,:,:])

    head = iradon(sino, angles, circle = False)
    a.imshow(head)
    a.set_title(dataset[1], fontsize = 6)
    a.set_axis_off()

    a2.imshow(im1-im2)
    a2.set_title(dataset[1], fontsize = 6)
    a2.set_axis_off()
    
fig.savefig(results_folder+'CheckWalls_Test44.pdf')
fig2.savefig(results_folder+'CheckWallsDifference_Test44.pdf')
