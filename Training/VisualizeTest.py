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
nLayer = 8 
epochs = 30
lam = 0.05
max_angle = 640
proj_num = 72
img_size = 100
train_size = 0.7
train_name = 'Test42_MODLNetwork'
batch_size = 5

with open(results_folder+train_name+'Proj{}_nLay{}_epochs{}_K{}_lam{}_trnSize{}.pkl'.format(proj_num, nLayer, epochs, K, lam, train_size), 'rb') as f:
    test_loss_total = pickle.load(f)

#print(test_loss_total)
print([np.array(arr).mean() for arr in test_loss_total[72].values()])
