#%%
'''
Processes datasets for different acceleration factors
author: obanmarcos
'''

from config import *
from concurrent.futures import process
import os, sys

# Where am I asks where you are
sys.path.append(where_am_i())

import seaborn as sns
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.ndimage as ndi
from utilities import dataloading_utilities as dlutils
from utilities.folders import *
from torch.utils.data import DataLoader, ConcatDataset
from skimage.transform import radon, iradon

def process_datasets(args_options):
    
    folder_paths = ['/home/obanmarcos/Balseiro/DeepOPT/DataOPT/140415_5dpf_4X']

    zebra_dataset_dict = {'dataset_folder':'/home/obanmarcos/Balseiro/DeepOPT/datasets/full_fish',
                          'experiment_name':'Bassi',
                          'img_resize' :100,
                          'load_shifts':False,
                          'save_shifts': True,
                          'number_projections_total':720,
                          'number_projections_undersampled': 72,
                          'batch_size': 5,
                          'sampling_method': 'equispaced-linear',
                          'acceleration_factor': 10}

    # 1 - Load datasets
    # 1a - Check ZebraDataset writing of x10 acceleration factor
    for acceleration_factor in args_options['acc_factors']:
        
        zebra_dataset_dict['acceleration_factor'] = acceleration_factor
        zebra_dataset_dict['number_projections_undersampled'] = zebra_dataset_dict['number_projections_total']//zebra_dataset_dict['acceleration_factor']

        for folder in folder_paths:
            
            zebra_dataset_dict['folder_path'] = folder
            zebra_dataset_test = dlutils.DatasetProcessor(zebra_dataset_dict)
            
            return zebra_dataset_test
        
#%%

args_options = {}
args_options['acc_factors'] = [20]


test_dataset = process_datasets(args_options)

#%%

test_dataset.load_images()
angles =  np.linspace(0, 2*180, test_dataset.image_volume.shape[1] ,endpoint = False)
#%%

def search_shifts(image, max_shift=100, shift_step= 10,center_shift_top=0, center_shift_bottom=0):

    # Sweep through all shifts
    top_shifts = np.arange(-max_shift, max_shift, shift_step)+center_shift_top
    bottom_shifts = np.arange(-max_shift, max_shift, shift_step)+center_shift_bottom
    angles =  np.linspace(0, 2*180, test_dataset.image_volume.shape[0] ,endpoint = False)

    top_image_std = []
    bottom_image_std = []

    for i, (top_shift, bottom_shift) in enumerate(zip(top_shifts, bottom_shifts)):

      print('Shift {}, top shift {}, bottom shift {}'.format(i, top_shift, bottom_shift))

      top_shift_sino = ndi.shift(image, (top_shift, 0), mode = 'nearest')

      # Get image reconstruction
      top_shift_iradon =  iradon(top_shift_sino, angles, circle = False)
      
      # Calculate variance
      top_image_std.append(np.std(top_shift_iradon))
    
    plt.plot(top_shifts, top_image_std)

    max_shift_top = top_shifts[np.argmax(top_image_std)]

    return (max_shift_top)

#%%

print(search_shifts(test_dataset.image_volume[:,:,300].T))
#%%

fig, axs = plt.subplots(1, 3, figsize = (6, 5))
axs = axs.flatten()

shifts = []

for i in np.arange(8, 16, 1):
    
    angles =  np.linspace(0, 2*180, test_dataset.image_volume.shape[0], endpoint = False)
    top_shift_sino = ndi.shift(test_dataset.image_volume[:,:,300], (0, i), mode = 'nearest')
    image = iradon(top_shift_sino.T, angles, circle=False)
    # cv2.rectangle(image,(0,0),(100,100),(0,0,0),10)
    
    # ax.imshow(image, cmap = 'magma')
    # ax.set_axis_off()
    image = (image - image.min())/(image.max() - image.min())
    # ax.set_title(f'+{i}px\n'+r'$\mu_I$ = '+f'{round(image.mean(), 3)}')
    shifts.append([i, round(image.mean(), 3), round(image.std(), 3)])
#%%
shifts = np.array(shifts)
fig, ax1 = plt.subplots()


ln1 = ax1.plot(shifts[:,0], shifts[:,1], label = r'$\sigma_I$')
ax2 = ax1.twinx()
ln2 = ax2.plot(shifts[:,0], -shifts[:,2]+shifts[:,2].max(), color = 'orange', label = r'$\mu_I$')

ax1.axvline(12, color = 'purple')
ax1.axvline(10, color = 'red', linestyle = '--')
ax1.axvline(14, color = 'red', linestyle = '--')
sns.despine()


ax1.grid(alpha =0.2)

lns = ln1+ln2
labs = [l.get_label() for l in lns]

ax1.legend(lns, labs)
ax1.set_ylabel(r'$\mu_I$ [a.u.]', fontsize= 14)
ax1.set_xlabel('Registration shift [pixels]', fontsize = 14)
ax2.set_ylabel(r'$\sigma_I$ [a.u.]', fontsize= 14)

    
fig.savefig('')    
