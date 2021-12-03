'''
Plot Test results for layers
'''
import pickle
import os, sys
os.chdir('/home/marcos/DeepOPT/')
sys.path.append('Utilities/')
sys.path.append('OPTmodl/')
import matplotlib.pyplot as plt
import numpy as np
from Folders_cluster import *
import ModelUtilities as modutils

K = 10
nLayer = 9 
epochs = 50
lam = 0.05
max_angle = 640
proj_num = 72
img_size = 100
train_size = 0.7
train_name = 'Optimization_Layers_Test29'
batch_size = 5

with open(results_folder+train_name+'Layers_Proj{}_nLay{}_K{}_lam{}_trnSize{}.pkl'.format(proj_num, nLayer, K, lam, train_size), 'rb') as f:
    test_loss_total = pickle.load(f)

#print(test_loss_total)

Layers_means = []
fbp_loss = []
for k,v in test_loss_total.items():
    
    Layers_means.append(np.mean(np.array(v['loss_net'])))
    fbp_loss.append(np.mean(np.array(v['loss_fbp'])))

print(Layers_means)
print(fbp_loss)
Layers_idx = np.arange(3,nLayer+1)

#%% Plot
fig, ax = plt.subplots(1,1, figsize = (8,6))

ax.plot(Layers_idx, Layers_means, 'r*-', label = 'MoDL test loss')
ax.plot(Layers_idx, fbp_loss, 'b*-',label = 'FBP test loss' )
ax.set_xlabel('# of layers', fontsize = 15)
ax.set_ylabel('PSNR [dB]', fontsize = 15)
ax.legend()
ax.grid(True)

fig.savefig(results_folder+train_name+'PlotLoss_Proj{}_nLay{}_epochs{}_K{}_lam{}_trnSize{}.pdf'.format(proj_num, nLayer, epochs, K, lam, train_size))
