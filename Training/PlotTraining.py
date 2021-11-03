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
lam = 0.05
max_angle = 720
proj_num = 72
img_size = 100
train_size = 100
train_name = 'NBN_Lambdas_Test15'
batch_size = 5

with open(results_folder+train_name+'Dict_Proj{}_nlay{}_epochs{}_K{}_lam{}_trnSize{}.pkl'.format(proj_num, nLayer, epochs, K, lam, train_size), 'rb') as f:
    train_info = pickle.load(f)
   
#%% Plot
fig, ax = plt.subplots(1,1, figsize = (8,6))

train = modutils.psnr(img_size, train_info[K]['train'], batch_size)
val = modutils.psnr(img_size,train_info[K]['val'], batch_size)
train_fbp = modutils.psnr(img_size,train_info[K]['train_fbp'], batch_size)

print(val[-1])
print(train_fbp[-1])
ax.plot(train, label='PSNR de entrenamiento')
ax.plot(val, label = 'PSNR de validación')
ax.plot(train_fbp, label = 'PSNR de referencia\nRetroproyección filtrada', linestyle = '--')
#ax.plot(modutils.psnr(img_size, train_info[K]['train_backproj'], batch_size), label = 'Backprojection loss benchmark', linestyle = '--')
#ax.plot(modutils.psnr(img_size, train_info[K]['train_norm'], batch_size), label = 'Normalization output and target')

ax.set_xlabel('Épocas', fontsize = 15)
ax.set_ylabel('PSNR [dB]', fontsize = 15)
ax.legend()
ax.grid(True)

fig.savefig(results_folder+train_name+'PlotLoss_Proj{}_nLay{}_epochs{}_K{}_lam{}_trnSize{}.pdf'.format(proj_num, nLayer, epochs, K, lam, train_size))
