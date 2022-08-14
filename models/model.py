"""
This code creates the model described in MoDL: Model-Based Deep Learning Architecture for Inverse Problems for OPT data, modifying the implementation for PyTorch in order to use Torch Radon

@author: obanmarcos
"""
import torch
import torch.nn as nn
import numpy as np
from torch_radon import Radon, RadonFanbeam
from torch_radon.solvers import cg
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
import torch.nn.functional as F
import monai

