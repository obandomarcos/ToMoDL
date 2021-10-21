import pickle
import os, sys
os.chdir('/home/marcos/DeepOPT/')
sys.path.append('Utilities/')
sys.path.append('OPTmodl/')
import matplotlib.pyplot as plt
import numpy as np
from Folders_cluster import *
import ModelUtilities as modutils

nLayer = 4
K = 1
epochs = 20
lam = 0.05
max_angle = 720
proj_num = 72
img_size = 100
train_size = 100
train_name = 'Lambdas'
batch_size = 5

with open(results_folder+train_name+'Dict_Proj{}_nlay{}_epochs{}_K{}_lam{}_trnSize{}.pkl'.format(proj_num, nLayer, epochs, K, lam, train_size), 'rb') as f:
    train_info = pickle.load(f)
   
#%% Plot
fig, ax = plt.subplots(1,1, figsize = (8,6))

ax.plot(modutils.psnr(img_size, train_info[K]['train'], batch_size), label='train loss')
ax.plot(modutils.psnr(img_size,train_info[K]['val'], batch_size), label = 'validation loss')
ax.plot(modutils.psnr(img_size,train_info[K]['train_fbp'], batch_size), label = 'FBP loss benchmark', linestyle = '--')
ax.set_xlabel('Epochs')
ax.set_ylabel('PSNR [dB]')
ax.legend()
ax.grid(True)

fig.savefig(results_folder+train_name+'PlotLoss_Proj{}_nLay{}_epochs{}_K{}_lam{}_trnSize{}.pdf'.format(proj_num, nLayer, epochs, K, lam, train_size))
