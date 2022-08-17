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
from torch.utils.data import DataLoader, ConcatDataset
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

    folder_paths = [f140115_1dpf, f140315_3dpf, f140419_5dpf, f140714_5dpf, f140117_3dpf, f140114_5dpf]
    
    folder_paths_names = ['f140115_1dpf', 'f140315_3dpf', 'f140419_5dpf', 'f140714_5dpf', 'f140117_3dpf', 'f140114_5dpf']

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
        
        for folder in folder_paths:
            
            zebra_dataset_dict['folder_path'] = folder
            zebra_dataset_test = dlutils.DatasetProcessor(zebra_dataset_dict)

            del zebra_dataset_test

    # Checking ZebraDataloaders
    if 'check_dataloader_building' in testing_options:
        
        acceleration_factor = 10

        folders_datasets = [datasets_folder+'/x{}/'.format(acceleration_factor)+x for x in os.listdir(datasets_folder+'x{}'.format(acceleration_factor))]

        for enum, folder in enumerate(folders_datasets):

            dataset_dict = {'root_folder' : folder, 'acceleration_factor' : acceleration_factor,
            'transform':None}
            
            dataset = dlutils.ReconstructionDataset(**dataset_dict) 
            
            dataloader = DataLoader(dataset, shuffle = True)

            (us_uf_img, us_fil_img, fs_fil_img) = next(iter(dataloader))
            
            us_uf_img = us_uf_img[0,...].cpu().detach().numpy()
            us_fil_img = us_fil_img[0,...].cpu().detach().numpy()
            fs_fil_img = fs_fil_img[0,...].cpu().detach().numpy()

            thumbs = cv2.imwrite(dataloader_testing_folder+'Test_Image_Dataloader_us_fil_imgs_uf_{}.jpg'.format(enum), 255.0*us_fil_img)
            
            thumbs = cv2.imwrite(dataloader_testing_folder+'Test_Image_Dataloader_us_uf_imgs_uf_{}.jpg'.format(enum), 255.0*us_uf_img)

            thumbs = cv2.imwrite(dataloader_testing_folder+'Test_Image_Dataloader_fs_fil_imgs_uf_{}.jpg'.format(enum), 255.0*fs_fil_img)       
            
    
    if 'check_multiple_dataset_dataloader_building' in testing_options:


        acceleration_factor = 10

        folders_datasets = [datasets_folder+'/x{}/'.format(acceleration_factor)+x for x in os.listdir(datasets_folder+'x{}'.format(acceleration_factor))]

        datasets = []

        for enum, folder in enumerate(folders_datasets):

            dataset_dict = {'root_folder' : folder, 'acceleration_factor' : acceleration_factor,
            'transform':None}
            
            datasets.append(dlutils.ReconstructionDataset(**dataset_dict))
        
        dataloader_test = DataLoader(ConcatDataset(datasets), shuffle = True)

        (us_uf_img, us_fil_img, fs_fil_img) = next(iter(dataloader_test))
            
        us_uf_img = us_uf_img[0,...].cpu().detach().numpy()
        us_fil_img = us_fil_img[0,...].cpu().detach().numpy()
        fs_fil_img = fs_fil_img[0,...].cpu().detach().numpy()

        thumbs = cv2.imwrite(dataloader_testing_folder+'Test_Image__Multidataset_Dataloader_us_fil_imgs_uf.jpg', 255.0*us_fil_img)
        
        thumbs = cv2.imwrite(dataloader_testing_folder+'Test_Image__Multidataset_Dataloader_us_uf_imgs_uf.jpg', 255.0*us_uf_img)

        thumbs = cv2.imwrite(dataloader_testing_folder+'Test_Image__Multidataset_Dataloader_fs_fil_imgs_uf.jpg', 255.0*fs_fil_img)  

        print(thumbs)

if __name__ == '__main__':

    testing_options = []

    parser = argparse.ArgumentParser(description='Test utilities.dataloading_utilities module')

    parser.add_argument('--dataset_writing', help = 'Test dataset loading', action="store_true")
    parser.add_argument('--dataset_registration', help = 'Test dataset registration', action="store_true")
    parser.add_argument('--dataloaders_building', help = 'Test dataloaders building', action="store_true")
    parser.add_argument('--multidataset_dataloader_building', help = 'Test multidataset dataloader building', action="store_true")
    
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
    
    if args.multidataset_dataloader_building:

        testing_options.append('check_multiple_dataset_dataloader_building')
    
    
    test_dataloader(testing_options)