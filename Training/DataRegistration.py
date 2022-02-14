'''
Registration of datasets
'''
#%%
import os
import os,time, sys
prefix_local = '/home/obanmarcos/Balseiro/Maestría/Proyecto/Implementación/'
os.chdir('/home/marcos/DeepOPT/')
sys.path.append('Utilities/')
sys.path.append('OPTmodl/')

import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

import DataLoading as DL
import torch
from torch_radon import Radon, RadonFanbeam
from skimage.transform import radon, iradon
import phantominator as ph
import torchvision
import model_torch as modl
import math
from Folders_cluster import *
import pathlib
import scipy.ndimage as ndi
import imageio
import cv2
#%%
folder_paths = [f140114_5dpf, f140117_3dpf, f140115_1dpf, f140315_3dpf, f140419_5dpf, f140714_5dpf]
samples = ['lower tail', 'upper tail', 'body', 'head']
images = []
slice_idx = 400
all_shifts = []

for folder_path in folder_paths:

    df = DL.ZebraDataset(folder_path, 'Datasets', 'Bassi')    
    
    print(df.fishPartsAvailable)

    for sample in df.fishPartsAvailable:
        
        df.loadImages(sample = sample)

        non_registered = np.copy(df.registeredVolume[sample][:,:,slice_idx])

        df.correctRotationAxis(sample = sample, max_shift = 200, shift_step = 1, load_shifts = True, save_shifts = False)
        
        angles = np.linspace(0, 2*180, df.registeredVolume[sample].shape[0] ,endpoint = False)

        all_shifts.append((str(df.folderName) , np.copy(df.shifts[sample]), np.copy(df.registeredVolume[sample][:,:,slice_idx]), np.copy(angles), np.copy(non_registered)))

fig_shift, ax_shift =plt.subplots(3, len(all_shifts)//3, figsize = (20, 20))
fig_images, ax_images = plt.subplots(3, len(all_shifts)//3, figsize = (20, 20))
fig_images_non_reg, ax_images_non_reg = plt.subplots(3, len(all_shifts)//3, figsize = (20, 20))

for (a_shift, a_images, a_images_non_reg, shift) in zip(ax_shift.flatten(), ax_images.flatten(), ax_images_non_reg.flatten(), all_shifts):

    a_shift.plot(shift[1])
    a_shift.set_title(shift[0])
    a_shift.set_xlabel('Slice')
    a_shift.set_ylabel('Pixel Shift')

    a_images.imshow(iradon(shift[2].T, shift[3], circle = False))
    a_images.set_axis_off()
    a_images.set_title(shift[0])

    a_images_non_reg.imshow(iradon(shift[4].T, shift[3], circle = False))
    a_images_non_reg.set_axis_off()
    a_images_non_reg.set_title(shift[0])

fig_shift.savefig(results_folder+'Test50_Shift_DataAxisCorrection.pdf', bbox_inches = 'tight')
fig_images.savefig(results_folder+'Test50_Images_DataAxisCorrection.pdf', bbox_inches = 'tight')
fig_images_non_reg.savefig(results_folder+'Test50_ImagesNonReg_DataAxisCorrection.pdf', bbox_inches = 'tight')
    
