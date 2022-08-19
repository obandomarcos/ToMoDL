'''
Processes datasets for different acceleration factors
author: obanmarcos
'''

from config import *
from concurrent.futures import process
import os, sys

sys.path.append(where_am_i())

import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utilities import dataloading_utilities as dlutils
from utilities.folders import *
from torch.utils.data import DataLoader, ConcatDataset

def process_datasets(args_options):
    
    folder_paths = [f140115_1dpf, f140315_3dpf, f140419_5dpf, f140714_5dpf, f140117_3dpf, f140114_5dpf]

    zebra_dataset_dict = {'dataset_folder':datasets_folder,
                          'experiment_name':'Bassi',
                          'img_resize' :100,
                          'load_shifts':True,
                          'save_shifts':False,
                          'number_projections_total':720,
                          'number_projections_undersampled': 72,
                          'batch_size': 5,
                          'sampling_method': 'equispaced-linear'}

    # 1 - Load datasets
    # 1a - Check ZebraDataset writing of x10 acceleration factor
    for acceleration_factor in args_options['acc_factors']:

        zebra_dataset_dict['number_projections_undersampled'] = zebra_dataset_dict['number_projections_undersampled']//acceleration_factor

        for folder in folder_paths:
            
            zebra_dataset_dict['folder_path'] = folder
            zebra_dataset_test = dlutils.DatasetProcessor(zebra_dataset_dict)

            del zebra_dataset_test

if __name__ == '__main__':
    
    args_options = np.arange(2, 30, 2).astype(int)

    process_datasets(args_options)