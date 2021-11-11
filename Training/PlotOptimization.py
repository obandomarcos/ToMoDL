import pickle
import os, sys
os.chdir('/home/marcos/DeepOPT/')
sys.path.append('Utilities/')
sys.path.append('OPTmodl/')
import matplotlib.pyplot as plt
import numpy as np
from Folders_cluster import *
import ModelUtilities as modutils
from scipy.interpolate import interp2d

train_name = 'OptimizationLambdaK_Test23'

with open(results_folder+'OptimizationValues_'+train_name+'.pkl', 'rb') as f:

    optimisation_info = pickle.load(f)
    
lambdas_name = 'lambdas'
K_name = 'K'
loss = []
lambdas = []
K = []

img_size = 100
batch_size = 5

for val in optimisation_info:
    
    if val['params'][lambdas_name]<7.5:    
        
        loss.append(10*np.log10(1.0*batch_size/(-val['target']/img_size**2)))
        lambdas.append(val['params'][lambdas_name])
        K.append(int(val['params']['K']))


loss_lambdas = np.vstack((K, lambdas, loss)).T
#loss_lambdas = loss_lambdas[loss_lambdas[:,1]<7.5]

#loss_lambdas = loss_lambdas[:,np.argsort(loss_lambdas[1,:])]
f = interp2d( lambdas,K, loss, kind = 'linear')
x_coords = np.arange(min(K)-1, max(K)+1)
y_coords = np.arange(min(lambdas)-1, max(lambdas)+1, 0.01)
Z = f(x_coords, y_coords)

print(loss_lambdas)
fig, ax = plt.subplots(1,1)

im = ax.imshow(Z, extent=[min(K)-1,max(K)+1,min(lambdas)-1,max(lambdas)+1],origin = 'lower', aspect = 'auto', interpolation= 'bilinear')
ax.scatter(x=K, y=lambdas, marker = 'x')
ax.set_xlabel('K')
ax.set_ylabel('Lambda')

cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])

cbar = plt.colorbar(im, cax = cax)
cbar.ax.get_yaxis().labelpad = 15
cbar.ax.set_ylabel("PSNR [dB]", rotation = 270)

fig.savefig(results_folder+'OptimizationValues_'+train_name+'.pdf', bbox_inches = 'tight')
