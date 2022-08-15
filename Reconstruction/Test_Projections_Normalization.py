"""
Test for normalization issues in reconstruction with TwIST and ADMM
author : obanmarcos
"""
#%%
import numpy as np
import scipy as sp
from scipy import io
import matplotlib.pyplot as plt
import phantominator as ph
from skimage.transform import radon, iradon
# import cupy as cp
from time import time
import sys
import pickle

sys.path.append('../Utilities/')
sys.path.append('../Reconstruction/')
%load_ext autoreload
%autoreload 1
%aimport Reconstruction
%aimport DataLoading 
%aimport ReconstructionHessian

rc = Reconstruction
dl = DataLoading
rh = ReconstructionHessian

from skimage.metrics import structural_similarity as ssim
from numpy.linalg import norm
import importlib

#%% Load dataset
