'''
Testing functionalities of models systems
author: obanmarcos
'''
import os
import os, sys

sys.path.append('/home/obanmarcos/Balseiro/DeepOPT/')

import pytorch_lightning as pl
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utilities import dataloading_utilities as dlutils
from utilities.folders import *
from utilities import model_utilities as modutils # Esto lo tengo que eliminar una vez que termine el models system
from models.models_system import MoDLReconstructor
import torch
import cv2

def test_pytorch_training(testing_options):
    '''
    Tests Dataloading.
    Checks dataset loading and Dataloader conformation
    Params:
        testing_options (list of string): testing options
    '''
    # Using CPU or GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device {}'.format(device))

    dataloader_testing_folder = '/home/obanmarcos/Balseiro/DeepOPT/Tests/Tests_Pytorch_Utilities/'

    folder_paths = [f140115_1dpf, f140315_3dpf, f140419_5dpf]

    # Dataloader parameters
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
    
    # ResNet dictionary parameters
    resnet_options_dict = {'number_layers': 3,
                           'kernel_size':3,
                           'features':64,
                           'in_channels':1,
                           'out_channels':1,
                           'stride':1, 
                           'use_batch_norm': True,
                           'init_method': 'xavier'}

    # Model parameters
    modl_dict = {'use_torch_radon': True,
                 'number_layers': 3,
                 'K_iterations' : 3,
                 'number_projections_total' : 720,
                 'number_projections_undersampled' : 72, 
                 'image_size': 100,
                 'lambda': 0.05,
                 'use_shared_weights': True,
                 'denoiser_method': 'resnet',
                 'resnet_options': resnet_options_dict,
                 'in_channels': 1,
                 'out_channels': 1}

    # Training parameters
    loss_dict = {'loss_name': torch.nn.MSELoss(reduction = 'sum')}
    optimizer_dict = {'optimizer_name': 'Adam',
                      'lr': 1e-3}
    
    # System parameters
    model_system_dict = {'optimizer_dict': optimizer_dict,
                        'kw_dictionary_modl': modl_dict,
                        'loss_dict': loss_dict,}
    
    # Load dataloaders
    zebra_dataloaders = dlutils.ZebraDataloader(zebra_dict)

    # 1 - Check training with Pytorch Lightning
    if 'check_pytorch_lightning_training' in testing_options:

        # Load and build dataloaders
        zebra_dataloaders.register_datasets()
        zebra_dataloaders.build_dataloaders()

        # model
        modl_reconstruction = MoDLReconstructor(model_system_dict)

        # train model
        trainer = pl.Trainer()
        trainer.fit(model=modl_reconstruction, train_dataloader=zebra_dataloaders.dataloaders['train'])
    
if __name__ == '__main__':

    testing_options = []

    parser = argparse.ArgumentParser(description='Test models.models_system module')

    parser.add_argument('--pl-train', help = 'Test Pytorch lightning training', action="store_true")

    args = parser.parse_args()

    if args.pl_train:

        print('Checking Pytorch lightning training')
        testing_options.append('check_pytorch_lightning_training')
    
    test_pytorch_training(testing_options)