'''
Plot Heatmap
'''
import sys, os
os.chdir('/home/obanmarcos/Balseiro/DeepOPT/')

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.interpolate import griddata
import scipy as sp


K_layers = pd.read_csv('./results/HyperParams.csv')

K_layers_psnr = np.array((K_layers.K_iterations, K_layers.number_layers, K_layers['val/loss_mean']))


N=100     
extent=(1,10,1,10)
xs,ys = np.mgrid[extent[0]:extent[1]:N, extent[2]:extent[3]:N]
xs = np.linspace(extent[0], extent[1], N)
ys = np.linspace(extent[1], extent[2], N)

resampled=sp.interpolate.interp2d(K_layers_psnr[0,:],K_layers_psnr[1,:],K_layers_psnr[2,:], kind = 'linear')

fig, ax = plt.subplots(1, 1, figsize = (8,6))

c = ax.imshow(resampled(xs, ys)[::-1, ], extent=extent, interpolation='nearest')
ax.scatter(K_layers_psnr[0,:],K_layers_psnr[1,:],c='r', marker = '*')

cbar = plt.colorbar(c)

cbar.ax.set_ylabel('PSNR [dB]')
ax.set_ylabel('Número de capas')
ax.set_xlabel(r'Iteraciones $K$')

fig.savefig('results/0-HyperParams.pdf')
