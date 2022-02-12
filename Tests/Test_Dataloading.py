'''
Testing for data loading 
1 - Test registration functions
2 - Test correct volume loading 
'''
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

import torch
from torch_radon import Radon, RadonFanbeam
from skimage.transform import radon, iradon
import phantominator as ph
import torchvision
import model_torch as modl
import math

f140114_5dpf = "/home/marcos/DeepOPT/DataOPT/140114_5dpf"  # 5 days post-fertilization
f140117_3dpf = "/home/marcos/DeepOPT/DataOPT/140117_3dpf"  # 3 days post-fertilization
f140115_1dpf = "/home/marcos/DeepOPT/DataOPT/140315_1dpf"  # 1 days post-fertilization

f140315_3dpf = "/home/marcos/DeepOPT/DataOPT/140315_3dpf"     # 3 days post-fertilization
f140415_5dpf_4X = "/home/marcos/DeepOPT/DataOPT/140415_5dpf_4X"  # 5 days post-fertilization
f140419_5dpf = "/home/marcos/DeepOPT/DataOPT/140519_5dpf"     # 5 days post-fertilization

f140714_5dpf = "/home/marcos/DeepOPT/DataOPT/140714_5dpf"
f140827_3dpf_4X = "/home/marcos/DeepOPT/DataOPT/140827_3dpf_4X"
f140827_5dpf_4X = '/home/marcos/DeepOPT/DataOPT/140827_5dpf_4X'

folder_paths = [f140115_1dpf, f140117_3dpf, f140114_5dpf, f140315_3dpf]
results_folder = '/home/marcos/DeepOPT/Resultados/'
model_folder = '/home/marcos/DeepOPT/Models/'

#%% Cargo un dataset
dataset = folder_paths

df = DL.ZebraDataset(dataset, 'Datasets', 'Bassi')
# Cargo el dataset
df.loadImages(sample = 'head')
# Cargo las registraciones correspondientes
df.loadRegTransforms()
# Aplico las transformaciones para este dataset 
df.applyRegistration(sample = 'head')

print()
#%% Chequeo si visualizo correctamente el sinograma
print(math.ceil(abs(df.meanDisplacement//2)))
test_volume = df.getRegisteredVolume('head', margin = math.ceil(abs(df.meanDisplacement//2)), useSegmented = True, saveDataset = False)
test_measurement = test_volume[:,:,100] # Random sinogram

fig, ax = plt.subplots(1,1)
ax.imshow(test_measurement)
fig.savefig(results_folder+'Test_measurement_same_reg.pdf')
