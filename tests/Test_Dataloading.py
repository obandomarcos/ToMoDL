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

    folder_path = f140115_1dpf

    zebra_dataset_dict = {'folder_path':folder_path,
                'dataset_folder':datasets_folder,
                 'experiment_name':'Bassi',
                 'img_resize' :100,
                 'load_shifts':True,
                 'save_shifts':False,
                  'number_projections_total':720,
                  'number_projections_undersampled': 72,
                  'batch_size': 5,
                  'sampling_method': 'equispaced-linear'}

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

    # 1 - Load datasets
    # 1a - Check ZebraDataset writing of x10 acceleration factor
    if 'check_dataset_writing' in testing_options:
        
        sample = 'head'
        
        for folder in folder_paths:
            
            zebra_dataset_dict['folder_path'] = folder
            zebra_dataset_test = dlutils.ZebraDataset(zebra_dataset_dict)

            for sample in zebra_dataset_test.fish_parts_available:
                zebra_dataset_test.load_images(sample)
                zebra_dataset_test.correct_rotation_axis(sample = sample, max_shift = 200, shift_step = 1)
                zebra_dataset_test.dataset_resize(sample)
                
                zebra_dataset_test.write_dataset_reconstruction(sample)

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

    parser.add_argument('--dataset_writing', help = 'Test dataset loading', action="store_true")
    parser.add_argument('--dataset_registration', help = 'Test dataset registration', action="store_true")
    parser.add_argument('--dataloaders_building', help = 'Test dataloaders building', action="store_true")

    args = parser.parse_args()

    if args.dataset_writing:

        print('Checking Dataset writing')
        testing_options.append('check_dataset_writing')
    
    if args.dataset_registration:

        print('Checking Dataset registration')
        testing_options.append('check_registering_dataset')
    
    if args.dataloaders_building:

        print('Checking Dataloaders building + masking datasets')
        testing_options.append('check_dataloaders_building')
    
    
    test_dataloader(testing_options)