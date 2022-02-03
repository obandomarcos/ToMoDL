'''
Testing for data loading 
1 - Test registration functions
2 - Test correct volume loading 
'''

#%%
import os
import os,time, sys
prefix_local = '/home/obanmarcos/Balseiro/Maestría/Proyecto/Implementación/'
os.chdir(prefix_local+'DeepOPT/')
sys.path.append('Utilities/')
sys.path.append('OPTmodl/')

import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

%load_ext autoreload
%autoreload 2
%aimport DataLoading
DL = DataLoading 

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
#%% Import folder paths
folder_paths = [f140114_5dpf]
sample = 'lower tail'
#%%
df = DL.ZebraDataset(folder_paths[0], 'Datasets', 'Bassi')

df.loadImages(sample = 'lower tail')
#%%
shifts = df.correctRotationAxis(sample = 'lower tail', max_shift = 200, shift_step = 4)

#%%
df._grabImageIndexes()
#%%
plt.plot(df.shifts)
#%% Grab non-empty image
df.imageVolume = (df.imageVolume-df.imageVolume.min())/(df.imageVolume.max()-df.imageVolume.min())
#%% para buscar el threshold, busco las lineas con varianza no nula
df._registerVolume(sample = sample)
#%%
shift_idx = 800
proj_prueba = df.imageVolume[:,:,shift_idx]
angles = df.angles
# proj_prueba = ndi.shift(proj_prueba, (-shifts[0], 0))
image_prueba = iradon(proj_prueba.T, angles, circle = False)

plt.imshow(image_prueba)

#%% Grab slice index
df = DL.ZebraDataset(f140117_3dpf, 'Datasets', 'Bassi')

df.loadImages(sample = 'lower tail')
# maximum
img_test = df.imageVolume.min(axis = 0)
img_test = (((img_test-img_test.min())/(img_test.max()-img_test.min()))*255.0).astype(np.uint8)
img_test = ndi.gaussian_filter(img_test,(11,11))
# plt.plot(img_test.std(axis = 0))
# plt.axhline(img_test.std(axis = 0))
#plt.plot(np.diff(ndi.gaussian_filter(img_test,(11,11)).mean(axis = 0)))

plt.plot(img_test.std(axis = 0))
plt.axhline(img_test.std(axis = 0).mean(axis = 0))

print(np.where(img_test.std(axis = 0)>50)[0][0],np.where(img_test.std(axis = 0)>50)[0][-1] )

#%% Pruebas preliminares
top = df.imageVolume[:,:, 1].T
bottom = df.imageVolume[:,:, -1].T

print(bottom.mean())
shift = -100
angles = np.linspace(0, 2*180, bottom.shape[1] ,endpoint = False)

bottom_shift = ndi.shift(bottom, (shift, 0))[:shift,:]
bottom_shift_iradon = iradon(bottom_shift, angles, circle = False)
plt.imshow(top)

#%%
shifts = np.arange(-200, 200, 10)
top = df.imageVolume[:,:,0 ].T
bottom = df.imageVolume[:,:, -1].T
images = []
image_std = []

for i, shift in enumerate(shifts):

    print('Shift {}, {}'.format(i, shift))
    
    top_shift = ndi.shift(top, (shift, 0), mode = 'nearest')

    top_shift_iradon =  iradon(top_shift, angles, circle = False)

    images.append((i, np.copy(top_shift_iradon)))
    
    image_std.append(np.std(top_shift_iradon))

#%%
print(shifts[np.argmax(image_std)])
plt.plot(shifts, image_std)
#%%
idx = 13
plt.imshow(images[idx][1])

#%% Top and bottom comparison
shifts = np.arange(-200, 200, 2)
top = df.imageVolume[:,:, 0].T
bottom = df.imageVolume[:,:, -1].T
angles = np.linspace(0, 2*180, top.shape[1] ,endpoint = False)
images = []
top_image_std = []
bottom_image_std = []

for i,shift in enumerate(shifts):

    print('Shift {}'.format(i))

    top_shift = ndi.shift(top, (shift, 0))
    bottom_shift = ndi.shift(bottom, (shift, 0))

    # crop zero intensity padding
    if shift > 0:
        top_shift = top_shift[shift:,:]
        bottom_shift = bottom_shift[shift:,:]
    
    else:
        top_shift = top_shift[:shift,:]
        bottom_shift = bottom_shift[:shift,:]
    
    top_shift_iradon =  iradon(top_shift, angles, circle = False)
    bottom_shift_iradon =  iradon(bottom_shift, angles, circle = False)

    top_image_std.append(np.std(top_shift_iradon))
    bottom_image_std.append(np.std(bottom_shift_iradon))


#%%
fig, ax = plt.subplots(1,1)

ax.plot(shifts, top_image_std, label = 'Top')
ax.plot(shifts, bottom_image_std, label = 'Bottom')
ax.legend()
ax.set_xlabel('Pixels')
ax.set_ylabel('Intensity variance')

fig.savefig(results_folder+'Test49_TopBottomRotationAxis.pdf', bbox_inches = 'tight')

#%%
max_shift_bottom = shifts[np.argmax(top_image_std)]


#%%

def 




#%%

for image in images:

    fig, ax = plt.subplots(1,2, figsize = (8, 8))

    ax[0].imshow(image[1])
    ax[0].axis('off')
    # ax[0].text(0, 400, shift)
    
    ax[1].plot(shifts, image_std)
    ax[1].axvline(shifts[image[0]], color = 'red')
    ax[1].set_xlabel('Pixels')
    ax[1].set_ylabel('Intensity variance')
    asp = np.diff(ax[1].get_xlim())[0] / np.diff(ax[1].get_ylim())[0]
    ax[1].set_aspect(asp)
    ax[1].set_title('Dataset {}, {}'.format('f140114_5dpf',sample))

    fig.savefig(results_folder+'CorrectAxisTest/'+str(image[0])+'.png', bbox_inches = 'tight')


#%%
filenames = [results_folder+'CorrectAxisTest/'+str(image[0])+'.png' for image in images]

images_list = []
for filename in filenames:
    images_list.append(imageio.imread(filename))

imageio.mimsave(results_folder+'CorrectAxisTest/AxisAlignment.gif', images_list)

#%%
