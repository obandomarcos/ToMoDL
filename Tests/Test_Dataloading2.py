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

    folder_paths = [f140115_1dpf, f140315_3dpf, f140419_5dpf]

    zebra_dict= {'folder_paths': folder_paths,
                'tensor_path': None,
                 'img_resize': 100,
                  'total_size' : 30,
                  'train_factor': 0.7,
                  'val_factor': 0.1, 
                  'test_factor': 0.2,
                  'augment_factor': 1,
                  'load_shifts': True,
                  'save_shifts': False,
                  'load_tensor': False,
                  'save_tensor': False,
                  'use_rand': True,
                  'k_fold_datasets': 1,
                  'number_projections_total':720,
                  'number_projections_undersample': 72,
                  'batch_size': 5,
                  'sampling_method': 'equispaced-linear'}

    zebra_dataloaders = dlutils.ZebraDataloader(zebra_dict)

    # 1 - Load datasets
    # 1a - Check ZebraDataset.load_images()
    if 'check_dataset_loading' in testing_options:

        zebra_dataset_test = zebra_dataloaders[folder_paths[0]]
        zebra_dataset_test.load_images()
        
        # Print main attributes after load_images
        print(zebra_dataset_test.registered_dataset)
        print(zebra_dataset_test.registered_volume['head'].max())
        print(zebra_dataset_test.image_volume['head'].max())

        # Print registered
        cv2.imwrite(dataloader_testing_folder+'Test_Image.jpg', 255.0*zebra_dataset_test._normalize_image(zebra_dataset_test.image_volume['head'][0,...]))

        # Print non-registered
        cv2.imwrite(dataloader_testing_folder+'Test_Image_Registered.jpg', 255.0*zebra_dataset_test._normalize_image(zebra_dataset_test.registered_volume['head'][0,...]))
    
    # 2 - Check ZebraDataset registering (already done, just loading shifts)
    if 'check_registering_dataset' in testing_options:
        
        zebra_dataloaders.register_datasets()

        print(zebra_dataloaders.datasets_registered)
    
    # Checking ZebraDataloaders
    if 'check_dataloaders_building' in testing_options:
        
        zebra_dataloaders.register_datasets()
        print(zebra_dataloaders.datasets_registered)
        zebra_dataloaders.build_dataloaders()
        print('Printing images from dataloaders')

        sets = ['train', 'val', 'test']
        puts = ['x', 'filt_x', 'y']
        
        for set_name in sets:
            
            tuple_image = zebra_dataloaders._get_next_from_dataloader(set_name)
            
            for put_idx, put_name in enumerate(puts):
                
                image = tuple_image[put_idx][0,0,...].cpu().detach().numpy()

                print(set_name, put_name)
                print('Image Intensity - set {} put {}\n'.format(set_name, put_name), 'Max', image.max(), 'Min: ',image.min())

                cv2.imwrite(dataloader_testing_folder+'Test_Image_Dataloader_{}_{}.jpg'.format(set_name, put_name), 255.0*image)

    
if __name__ == '__main__':

    testing_options = []

    parser = argparse.ArgumentParser(description='Test utilities.dataloading_utilities module')

    parser.add_argument('--dataset_loading', help = 'Test dataset loading', action="store_true")
    parser.add_argument('--dataset_registration', help = 'Test dataset registration', action="store_true")
    parser.add_argument('--dataloaders_building', help = 'Test dataloaders building', action="store_true")

    args = parser.parse_args()

    if args.dataset_loading:

        print('Checking Dataset loading')
        testing_options.append('check_dataset_loading')
    
    if args.dataset_registration:

        print('Checking Dataset registration')
        testing_options.append('check_registering_dataset')
    
    if args.dataloaders_building:

        print('Checking Dataloaders building + masking datasets')
        testing_options.append('check_dataloaders_building')
    
    
    test_dataloader(testing_options)