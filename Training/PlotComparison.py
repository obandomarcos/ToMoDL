"""
Plot comparison of images with previously saved models
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
proj_num = 72

train_factor = 0.7
val_factor = 0.2
test_factor = 0.1
total_size = 3000

batch_size = 5
img_size = 100
augment_factor = 10
train_infos = {}
test_loss_dict = {}
tensor_path = datasets_folder+'Proj_{}_augmentFactor_{}_totalSize_{}_'.format(proj_num, augment_factor, total_size)

datasets = modutils.formRegDatasets(folder_paths, umbral_reg, img_resize = img_size)
datasets = []

dataloaders = modutils.formDataloaders(datasets, proj_num, total_size, train_factor, val_factor, test_factor, batch_size, img_size, tensor_path, augment_factor, load_tensor = True, save_tensor = False)

lam = 0.05
nLayer= 9
K = 10
epochs = 30
max_angle = 640

train_name = 'Optimization_Layers_Test29'
model_name = model_folder+train_name+'K_{}_lam_{}_nlay_{}_proj_{}'.format(K, lam, nLayer, proj_num)

model = modl.OPTmodl(nLayer, K, max_angle, proj_num, img_size, None, lam, True, results_folder)

modutils.load_net(model_name, model, device)
    
fig, ax = plt.subplots(3,4) 
    
for i, (inp, target, filtX) in enumerate(zip(dataloaders['test']['x'], dataloaders['test']['y'], dataloaders['test']['filtX'])): 
        
    if i == 3:
        break
    t1 = time.time()
    pred = model(inp)
    t2 = time.time()-t1

    ax[i,0].imshow(inp[0,0,...].detach().cpu().numpy(), cmap = 'RdGy')
    ax[i,1].imshow(pred['dc'+str(K)][0,0,...].detach().cpu().numpy(), cmap = 'RdGy')
    ax[i,2].imshow(target[0,0,...].detach().cpu().numpy(), 'RdGy')
    ax[i,3].imshow(filtX[0,0,...].detach().cpu().numpy(), 'RdGy')
    
    for a in ax[i,...]:
        
        a.set_axis_off()

print('Tiempo por slice', t2)

ax[0,0].set_title('Input')
ax[0,1].set_title('MoDL')
ax[0,2].set_title('Target')
ax[0,3].set_title('FBP\n reconstruction')

fig.savefig(results_folder+train_name+'PlotImagesTest.pdf', bbox_inches = 'tight')

