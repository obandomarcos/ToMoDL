import pickle
import os, sys
os.chdir('/home/marcos/DeepOPT/')
sys.path.append('Utilities/')
sys.path.append('OPTmodl/')
import matplotlib.pyplot as plt
import numpy as np
from Folders_cluster import *
import ModelUtilities as modutils
nLayer = 3
K = 5
epochs = 40
lam = 0.05
max_angle = 720
proj_num = 72
img_size = 100
train_size = 150

with open(results_folder+'FBP_error_projections_Proj{}_nlay{}_epochs{}_K{}_lam{}_trnSize{}.pkl'.format(proj_num, nLayer, epochs, K, lam, train_size), 'rb') as f:
    train_info = pickle.load(f)
   
#%% Plot
fig, ax = plt.subplots(1,1, figsize = (8,6))

ax.plot(modutils.psnr(img_size, train_info[proj_num]['train']), label='train loss')
ax.plot(modutils.psnr(img_size,train_info[proj_num]['val']), label = 'validation loss')
ax.plot(modutils.psnr(img_size,train_info[proj_num]['train_fbp']), label = 'FBP loss benchmark', linestyle = '--')
ax.set_xlabel('Epochs')
ax.set_ylabel('PSNR [dB]')
ax.legend()
ax.grid(True)

fig.savefig(results_folder+'PlotLoss_Proj{}_nLay{}_epochs{}_K{}_lam{}_trnSize{}.pdf'.format(proj_num, nLayer, epochs, K, lam, train_size))
