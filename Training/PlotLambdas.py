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
K = 10
epochs = 20
lambdas = [0.05, 0.1, 10, 50, 100, 500]
max_angle = 640
proj_num = 72
img_size = 100
train_size = 100
batch_size = 5
train_name = 'Lambdas'

train_info = {}

for lam in lambdas:

    with open(results_folder+train_name+'Dict_Proj{}_nlay{}_epochs{}_K{}_lam{}_trnSize{}.pkl'.format(proj_num, nLayer, epochs, K, lam, train_size), 'rb') as f:
        
        train_info[lam] = pickle.load(f)
K = 10

#%% Plot
fig, axs = plt.subplots(2,3, figsize = (8,6), sharex = True, sharey = True)

for (t_info_key, t_info), ax in zip(train_info.items(), axs.flatten()):
    print(t_info[K].keys())
    ax.plot(modutils.psnr(img_size, t_info[K]['train'], batch_size), label='train loss')
    ax.plot(modutils.psnr(img_size,t_info[K]['val'], batch_size), label = 'validation loss')
    ax.plot(modutils.psnr(img_size, t_info[K]['train_fbp'], batch_size), label = 'FBP loss benchmark', linestyle = '--')
    if t_info_key == 100:
       ax.set_xlabel('Epochs')
    ax.set_ylabel('PSNR [dB]')
    ax.set_title('Lambda {}'.format(t_info_key))
    if t_info_key == 0.1:
        ax.legend(fontsize = 'x-small', loc = 'lower center')
    ax.grid(True)

fig.savefig(results_folder+train_name+'PlotLoss_Proj{}_nLay{}_epochs{}_K{}_trnSize{}.pdf'.format(proj_num, nLayer, epochs, K, train_size), bbox_inches = 'tight')
