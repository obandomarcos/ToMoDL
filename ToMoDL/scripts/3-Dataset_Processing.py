'''
Processes datasets for different acceleration factors
author: obanmarcos
'''

from config import *
from concurrent.futures import process
import os, sys

# Where am I asks where you are
sys.path.append(where_am_i())

import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utilities import dataloading_utilities as dlutils
from utilities.folders import *
from torch.utils.data import DataLoader, ConcatDataset

def process_datasets(args_options):

    folder_paths = ["/home/nhattm/ToMoDL/datasets/DataOPT/140415_5dpf_4X"]

    zebra_dataset_dict = {
        "dataset_folder": "/home/nhattm/ToMoDL/datasets/full_fish",
        "experiment_name": "Bassi",
        "img_resize": 100,
        "load_shifts": False,
        "save_shifts": True,
        "number_projections_total": 720,
        "number_projections_undersampled": 72,
        "batch_size": 5,
        "sampling_method": "equispaced-linear",
        "acceleration_factor": 10,
    }

    # 1 - Load datasets
    # 1a - Check ZebraDataset writing of x10 acceleration factor
    for acceleration_factor in args_options['acc_factors']:

        zebra_dataset_dict['acceleration_factor'] = acceleration_factor
        zebra_dataset_dict['number_projections_undersampled'] = zebra_dataset_dict['number_projections_total']//zebra_dataset_dict['acceleration_factor']

        for folder in folder_paths:

            zebra_dataset_dict['folder_path'] = folder
            zebra_dataset_test = dlutils.DatasetProcessor(zebra_dataset_dict)

            del zebra_dataset_test

if __name__ == '__main__':
    
    '''
    To-Do: ArgParsing
    '''
    args_options = {}
    args_options['acc_factors'] = [20]

    process_datasets(args_options)
