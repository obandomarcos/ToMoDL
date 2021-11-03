import pickle
import os, sys
os.chdir('/home/marcos/DeepOPT/')
sys.path.append('Utilities/')
sys.path.append('OPTmodl/')
import matplotlib.pyplot as plt
import numpy as np
from Folders_cluster import *
import ModelUtilities as modutils

train_name = 'OptimizationLambda_Test14'

with open(results_folder+'OptimizationValues_'+train_name+'.pkl', 'rb') as f:

    optimisation_info = pickle.load(f)
    
parameter_name = 'lambdas'
loss = []
lambdas = []
img_size = 100
batch_size = 5

for val in optimisation_info:

    loss.append(-val['target'])
    lambdas.append(val['params'][parameter_name])

loss_lambdas = np.vstack((loss, lambdas))
loss_lambdas = loss_lambdas[:,np.argsort(loss_lambdas[1,:])]

print(loss_lambdas)
fig, ax = plt.subplots(1,1)

ax.plot(loss_lambdas[1,:], modutils.psnr(img_size, loss_lambdas[0,:], batch_size), linestyle = '-', marker = '*', label = 'Lambda')
ax.set_xlabel('Lambda')
ax.set_ylabel('PSNR [dB]')
ax.grid(True)

fig.savefig(results_folder+'OptmisationValues_'+train_name+'.pdf', bbox_inches = 'tight')
