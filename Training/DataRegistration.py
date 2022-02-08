'''
Registration of datasets
'''
#%%
import os
import os,time, sys
prefix_local = '/home/obanmarcos/Balseiro/Maestría/Proyecto/Implementación/'
os.chdir(prefix_local+'DeepOPT/')
sys.path.append('Utilities/')
sys.path.append('OPTmodl/')

import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

%load_ext autoreload
%autoreload 2
%aimport DataLoading
DL = DataLoading 

import torch
from torch_radon import Radon, RadonFanbeam
from skimage.transform import radon, iradon
import phantominator as ph
import torchvision
import model_torch as modl
import math
from Folders_cluster import *
import pathlib
import scipy.ndimage as ndi
import imageio
import cv2
#%%
folder_paths = [f140114_5dpf, f140117_3dpf, f140115_1dpf, f140315_3dpf, f140419_5dpf, f140714_5dpf]
samples = ['lower tail', 'upper tail', 'body', 'head']
images = []
slice_idx = 400

for folder_path in folder_paths:

    df = DL.ZebraDataset(folder_path, 'Datasets', 'Bassi')    
    
    print(df.fishPartsAvailable)

    for sample in df.fishPartsAvailable:
        
        df.loadImages(sample = sample)
        df.correctRotationAxis(sample = sample, max_shift = 200, shift_step = 1, load_shifts = False, save_shifts = True)