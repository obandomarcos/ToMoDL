'''
Load K-folding models from groups and evaluate performance

author : obanmarcos
'''
import os, sys
from config import * 

sys.path.append(where_am_i())

import lightning as pl
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utilities import dataloading_utilities as dlutils
from utilities.folders import *

from training import train_utilities as trutils

from models.models_system import MoDLReconstructor
import torch

from lightning.callbacks import ModelCheckpoint
from lightning.loggers import WandbLogger

from torch.utils.data import DataLoader
from torchvision import transforms as T
from pytorch_msssim import SSIM
# from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure as MSSSIM
import wandb
from pathlib import Path
import pandas as pd
import cv2 
import scipy.ndimage as ndi

def get_mask(img, thresh = 70):

    mask = np.zeros_like(img).astype(np.float64)
    mask[img>thresh] = 1

    mask = ndi.binary_closing(mask, iterations = 5).astype(np.float64)
    mask = (mask - mask.min())/(mask.max()-mask.min())

    return mask
def norm(img):

    return (img-img.mean())/img.std()
def read_volume(vol_path, use_mask = True):

    imgs_paths = [vol_path+'/'+path for path in os.listdir(vol_path)]

    imgs_vol = [norm(cv2.imread(img, cv2.COLOR_BGR2GRAY)) for img in imgs_paths if 'jpg' in img]
    if use_mask == True:

        masks = [get_mask(img) for img in imgs_vol]
        imgs_vol = [np.multiply(img_vol, mask/len(imgs_vol)) for img_vol, mask in zip(imgs_vol, masks)]

    return np.array(imgs_vol)

def norm_01(img):
    return img

if __name__ == '__main__':

    method = 'SSIM-Corte'
    imgs_paths = '/home/obanmarcos/Balseiro/DeepOPT/Volumes/MODL_SSIM_X10_K8_nLayers8'
    
    volume = read_volume(imgs_paths, False)
    N = volume.shape[2]
    vol_proj = volume[:, :, N//2-20:N//2+20].mean(axis = 2)
    
    fig, ax = plt.subplots(1, 1, figsize = (8,6))

    c = ax.imshow(vol_proj, aspect = 'equal', cmap = 'gray')

    fig.savefig('results/18-ReconstructFull_{}.pdf'.format(method))

    cv2.imwrite('results/18-ReconstructFull+{}.jpg'.format(method), vol_proj)
