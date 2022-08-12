'''
Testing functionalities of dataloading_utilities

author: obanmarcos
'''
import os
import os, sys

sys.path.append('/home/obanmarcos/Balseiro/DeepOPT/')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utilities import dataloading_utilities as dlutils
from utilities.folders import *
from utilities import model_utilities as modutils
import torch


# Using CPU or GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
folder_paths = [f140115_1dpf]



for folder_path in folder_paths:

    zebra_dataset = dlutils.ZebraDataset(folder_path, 'Datasets', 'Bassi')

    print(zebra_dataset._search_all_files(zebra_dataset.folder_path))
