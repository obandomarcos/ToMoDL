"""
Test how to register all the datasets I have
"""
#%%
import os
import os,time, sys
os.chdir('/home/obanmarcos/Balseiro/Maestría/Proyecto/Implementación/DeepOPT/')
sys.path.append('./Utilities/')
sys.path.append('./OPTmodl/')
import numpy as np
import time
import copy 
import datetime
import sys, os
from torch_radon import Radon, RadonFanbeam
import DataLoading as DL
import math
import matplotlib.pyplot as plt
import cv2 
from Folders_cluster import *
import resource

#%% Memory usage

def memory_limit():
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (get_memory() * 1024 *7/ 8, hard))

def get_memory():
    with open('/proc/meminfo', 'r') as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                free_memory += int(sline[1])
    return free_memory

#%% 
memory_limit()    
#%%
folder_paths = [f140117_3dpf, f140114_5dpf, f140315_3dpf, f140419_5dpf, f140115_1dpf,f140714_5dpf]

datasets = DL.ZebraDataset(folder_paths[0], 'Datasets', 'Bassi')
datasets.loadImages(sample = None)

datasets.dataset.head(5)

#%% Cargo las registraciones

datasets.loadRegTransforms()

#%%
datasets.applyRegistration()

