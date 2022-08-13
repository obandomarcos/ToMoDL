'''
Testing functionalities of dataloading_utilities

author: obanmarcos
'''
import os
import os, sys

sys.path.append('/home/obanmarcos/Balseiro/DeepOPT/')

import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utilities import dataloading_utilities as dlutils
from utilities.folders import *
from utilities import model_utilities as modutils
import torch
import cv2

def test_dataloader(testing_options):
    '''
    Tests Dataloading.
    Checks dataset loading and Dataloader conformation
    Params:
        testing_options (list of string): testing options
    '''
    # Using CPU or GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print('Device {}'.format(device))
    dataloader_testing_folder = '/home/obanmarcos/Balseiro/DeepOPT/Tests/Tests_Dataloader_Utilities/'

    folder_paths = [f140115_1dpf]

    zebra_dataloader = dlutils.ZebraDataloader(folder_paths, img_resize = 100)

    # 1 - Load datasets
    # 1a - Check ZebraDataset.load_images()
    if 'check_dataset_loading' in testing_options:

        zebra_dataset_test = zebra_dataloader[folder_paths[0]]
        zebra_dataset_test.load_images()
        
        # Print main attributes after load_images
        print(zebra_dataset_test.registered_dataset)
        print(zebra_dataset_test.registered_volume['head'].max())
        print(zebra_dataset_test.image_volume['head'].max())

        # Print registered
        cv2.imwrite(dataloader_testing_folder+'Test_Image.jpg', 255.0*zebra_dataset_test._normalize_image(zebra_dataset_test.image_volume['head'][0,...]))

        # Print non-registered
        cv2.imwrite(dataloader_testing_folder+'Test_Image_Registered.jpg', 255.0*zebra_dataset_test._normalize_image(zebra_dataset_test.registered_volume['head'][0,...]))

    if 'check_registering_dataset':
        
        datasets_registered = zebra_dataloader.register_datasets(number_projections = 720, load_shifts=True, save_shifts = False)

        print(datasets_registered)

if __name__ == '__main__':

    testing_options = []

    parser = argparse.ArgumentParser(description='Test utilities.dataloading_utilities module')

    parser.add_argument('--dataset_loading', help = 'Test dataset loading', action="store_true")
    parser.add_argument('--dataset_registration', help = 'Test dataset registration', action="store_true")
    
    args = parser.parse_args()

    if args.dataset_loading:

        print('Checking Dataset loading')
        testing_options.append('check_dataset_loading')
    
    if args.dataset_registration:

        print('Checking Dataset registration')
        testing_options.append('check_registering_dataset')
    
    test_dataloader(testing_options)