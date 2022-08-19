'''
Create main object for general training - Solve K-Folding
author: obanmarcos
'''


'''
Testing functionalities of dataloading_utilities

author: obanmarcos
'''
import os
import os, sys
from re import S
import names

sys.path.append('~/DeepOPT/')

import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utilities import dataloading_utilities as dlutils
from utilities.folders import *

from models import models_system as modsys

from torch.utils.data import DataLoader, ConcatDataset, random_split

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb

class TrainerSystem():

    def __init__(self, trainer_kwdict, dataloader_kwdict, model_system_kwdict):
        '''
        Trainer for MoDL.
        Params:
            - dataloader_kwdict (dict): Dictionary for dataloader creation
            - model_kwdict (dict): Dictionary for model creation
        '''
        self.process_trainer_kwdictionary(trainer_kwdict)
        self.process_dataloader_kwdictionary(dataloader_kwdict)
        self.process_model_system_kwdictionary(model_system_kwdict)
    
    def process_trainer_kwdictionary(self, kwdict):
        '''
        Process kwdictionary for training parameters
        Params:
            kwdict (dict): Dictionary for trainer init
        '''

        self.use_k_folding = kwdict['use_k_folding']
        
        if self.use_k_folding == True:
            
            # Number of datasets for testing
            self.k_fold_number_datasets = kwdict['k_fold_number_datasets']
            self.current_fold = 0 
        
        self.use_logger = kwdict['use_logger']
        self.lightning_trainer_dict = kwdict['lightning_trainer_dict']

        if self.use_logger == True:

            self.track_default_checkpoints = True
            self.logger_dict = kwdict['logger_dict']

            self.run_base_name = names.get_last_name()
        
        self.create_trainer()
            
    def reinitialize_logger(self):
        '''
        Reinitializes logger, meant for reloading K-folding
        
        '''
        if self.use_logger == True:              
            
            if self.use_k_folding == True:

                self.logger_dict['group'] = self.run_base_name
                self.logger_dict['name'] = 'K-Fold {}/{}'.format(self.current_fold, self.k_fold_max-1) 

            # Logger parameters
            self.wandb_logger = WandbLogger(**self.logger_dict) 

            # Callbacks (default)
            if self.track_default_checkpoints == True:

                train_psnr_fbp_checkpoint_callback = ModelCheckpoint(monitor='train/psnr_fbp', mode='max')
                train_ssim_fbp_checkpoint_callback = ModelCheckpoint(monitor='train/ssim_fbp', mode='max')

                val_psnr_fbp_checkpoint_callback = ModelCheckpoint(monitor='val/psnr_fbp', mode='max')
                val_ssim_fbp_checkpoint_callback = ModelCheckpoint(monitor='val/ssim_fbp', mode='max')

                train_psnr_checkpoint_callback = ModelCheckpoint(monitor='train/psnr', mode='max')
                train_ssim_checkpoint_callback = ModelCheckpoint(monitor='train/ssim', mode='max')

                val_psnr_checkpoint_callback = ModelCheckpoint(monitor='val/psnr', mode='max')
                val_ssim_checkpoint_callback = ModelCheckpoint(monitor='val/ssim', mode='max')

                self.lightning_trainer_dict['callbacks'] = [train_psnr_fbp_checkpoint_callback,
                                                            train_ssim_fbp_checkpoint_callback,
                                                            val_psnr_fbp_checkpoint_callback,
                                                            val_ssim_fbp_checkpoint_callback,
                                                            train_psnr_checkpoint_callback,
                                                            train_ssim_checkpoint_callback,
                                                            val_psnr_checkpoint_callback,
                                                            val_ssim_checkpoint_callback]
            
            self.lightning_trainer_dict['logger'] = self.wandb_logger
         
    def create_trainer(self):
        '''
        Create trainer based on current trainer_dict
        '''

        self.trainer = pl.Trainer(**self.lightning_trainer_dict)

        return self.trainer

    def process_dataloader_kwdictionary(self, kwdict):
        '''
        Process kwdictionary and creates dataloader
        Params:
            kwdict (dict): Dictionary for dataloader creation
        '''

        self.datasets_folder = kwdict['datasets_folder']
        self.experiment_name = kwdict['experiment_name']
        self.img_resize = kwdict['img_resize'] 
        self.load_shifts = kwdict['load_shifts']
        self.save_shifts = kwdict['save_shifts']                  
        self.number_projections_total = kwdict['number_projections_total']
        self.number_projections_undersampled = kwdict['number_projections_undersampled']
        
        # Dataset splitting (fraction)
        self.train_factor = kwdict['train_factor']
        self.val_factor = kwdict['val_factor']
        self.test_factor = kwdict['test_factor']

        self.batch_size = kwdict['batch_size']
        self.acceleration_factor = self.number_projections_total//self.number_projections_undersampled
        self.sampling_method= kwdict['sampling_method']
        self.shuffle_data = kwdict['shuffle_data']

        self.data_transform = kwdict['data_transform']
        
        self.number_volumes = kwdict['number_volumes']
        
        # To-Do: Option for non-available acceleration factors (RUN ProcessDatasets)
        self.folders_datasets = [self.datasets_folder+'x{}/'.format(self.acceleration_factor)+x for x in os.listdir(self.datasets_folder+'x{}'.format(self.acceleration_factor))]

        if self.number_volumes != 0:

            self.folders_datasets = self.folders_datasets[:self.number_volumes]
            
        # Number of datasets defines splitting number between train/val and test datasets
        if self.use_k_folding == True:
            
            self.datasets_number = len(self.folders_datasets)

            self.k_fold_max = self.datasets_number//self.k_fold_number_datasets
            
    def process_model_system_kwdictionary(self, kwdict):
        '''
        Process kwdictionary and creates dataloader
        Params:
            kwdict (dict): Dictionary for model creation
        '''

        self.model_system_dict = kwdict
        self.model_system_dict['max_epochs'] = self.lightning_trainer_dict['max_epochs']
    
    def generate_K_folding_dataloader(self):
        '''
        Rotates self.folders_datasets and builds new train/val/test dataloaders
        '''

        if self.current_fold != 0:
            # Rotate datasets
            self.rotate_list(self.folders_datasets, self.k_fold_number_datasets)

        # Load each dataset in Dataset class (torch.utils.data.Dataset)
        train_val_datasets_folders = self.folders_datasets[:self.datasets_number-self.k_fold_number_datasets].copy()
        test_datasets_folders = self.folders_datasets[self.datasets_number-self.k_fold_number_datasets:].copy()

        print('Train/Val folders in use...')
        print(train_val_datasets_folders)

        print('Test folders in use...')
        print(test_datasets_folders)

        # Train and validation dataloader  
        train_val_datasets = []
        
        for enum, folder in enumerate(train_val_datasets_folders):
            
            dataset_dict = {'root_folder' : folder, 
                            'acceleration_factor' : self.acceleration_factor,
                            'transform' : self.data_transform}

            train_val_datasets.append(dlutils.ReconstructionDataset(**dataset_dict))
        
        train_val_datasets = ConcatDataset(train_val_datasets)
        
        train_val_lengths = [int(len(train_val_datasets)*self.train_factor), int(len(train_val_datasets)*self.val_factor)]
        
        # Possible non-zero sum
        if sum(train_val_lengths) != len(train_val_datasets):

            train_val_lengths[0] += (len(train_val_datasets)- sum(train_val_lengths))

        train_dataset, val_dataset = random_split(train_val_datasets, train_val_lengths)

        train_dataloader = DataLoader(train_dataset, 
                                batch_size = self.batch_size,
                                shuffle = self.shuffle_data,
                                num_workers = 16)

        val_dataloader = DataLoader(val_dataset, 
                                batch_size = self.batch_size,
                                shuffle = False,
                                num_workers = 16)
        
        test_datasets = []
        
        for enum, folder in enumerate(test_datasets_folders):
            
            dataset_dict = {'root_folder' : folder, 
                                'acceleration_factor' : self.acceleration_factor,
                                'transform' : self.data_transform}

            test_datasets.append(dlutils.ReconstructionDataset(**dataset_dict))
        
        test_dataset = ConcatDataset(test_datasets)

        test_dataloader = DataLoader(test_dataset, 
                                batch_size = self.batch_size,
                                shuffle = False,
                                num_workers = 16)

        return train_dataloader, val_dataloader, test_dataloader
    
    @staticmethod
    def rotate_list(lst, n, direction = 'backwards'):
        '''
        Rotate list n-steps in direction 
        '''
        if direction == "backwards":
            for _ in range(0,n):
                lst.append(lst.pop(0))
        else: 
            for _ in range(0,n):
                lst.insert(0,lst.pop())
        return lst

    def train_model(self):
        '''
        Train model based on Pytorch Lightning
        '''
        self.reinitialize_logger()
        # Create dataloaders
        train_dataloader, val_dataloader, test_dataloader = self.generate_K_folding_dataloader()

        # PL train model + Update wandb
        trainer = self.create_trainer()
        
        # Create model reconstructor
        modl_reconstruction = modsys.MoDLReconstructor(self.model_system_dict)

        modl_reconstruction.log('k_fold', self.current_fold)
        # W&B logger
        self.wandb_logger.watch(modl_reconstruction)

        # Train model
        trainer.fit(model=modl_reconstruction, 
                    train_dataloaders= train_dataloader,
                    val_dataloaders = val_dataloader)

        # test model
        trainer.test(model = modl_reconstruction, dataloaders = test_dataloader)

        self.wandb_logger.finalize('success')

        wandb.finish()

    def k_folding(self):
        '''
        K-fold with parameters
        '''
        print('Running K-Folding...')
        assert(self.use_k_folding == True)

        for k_fold in range(self.k_fold_max):
            
            print('{} fold started...'.format(self.current_fold))
            self.train_model()
            print('{} fold finished succesfully!'.format(self.current_fold))
            self.current_fold += 1
            

        