'''
This code provides with an easy way to avoid boilerplate for training and validating results.
author: obanmarcos
'''

import torch
from . import modl
import pytorch_lightning as pl
import torch.nn as nn
import numpy as np
from torch_radon import Radon, RadonFanbeam
from torch_radon.solvers import cg
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import unet
import wandb 

# Modify for multi-gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MoDLReconstructor(pl.LightningModule):
    '''
    Pytorch Lightning for MoDL boilerplate
    '''
    def __init__(self, kw_dictionary_model_system):
        '''
        Initializes MoDL reconstructor. 
        Params:
            - kw_dictionary_modl (dict): 
        '''
        super().__init__()
        
        wandb.init(project = 'deepopt')

        self.process_kwdictionary(kw_dictionary_model_system)

        self.model = modl.modl(self.kw_dictionary_modl)

        self.save_hyperparameters()

    def forward(self, x):

        return self.model(x)['dc'+str(self.model.K)]
 
    def training_step(self, batch, batch_idx):
        '''
        Training step for modl. 
        Suffixes:
            - 'us' stands for undersampled reconstruction (used as input with unfiltered backprojection)
            - 'fs' stands for fully sampled reconstruction
        '''

        unfiltered_us_rec, filtered_us_rec, filtered_fs_rec = batch
        
        modl_rec = self.model(unfiltered_us_rec)

        psnr_fbp_loss = self.loss_dict['psnr_loss'](filtered_us_rec, filtered_fs_rec)
        ssim_fbp_loss = self.loss_dict['ssim_loss'](filtered_us_rec, filtered_fs_rec)

        self.log("train/psnr_fbp", self.psnr(psnr_fbp_loss))
        self.log("train/ssim_fbp", ssim_fbp_loss)

        psnr_loss = self.loss_dict['psnr_loss'](modl_rec['dc'+str(self.model.K)], filtered_fs_rec)
        ssim_loss = self.loss_dict['ssim_loss'](modl_rec['dc'+str(self.model.K)], filtered_fs_rec)

        self.log("train/psnr", self.psnr(psnr_loss))
        self.log("train/ssim", ssim_loss)

        if self.loss_dict['loss_name'] == 'psnr':
            
            return psnr_loss
        
        elif self.loss_dict['loss_name'] == 'ssim':
            
            return ssim_loss
    
    def validation_step(self, batch, batch_idx):

        '''
        Validation step for modl. 
        Suffixes:
            - 'us' stands for undersampled reconstruction (used as input with unfiltered backprojection)
            - 'fs' stands for fully sampled reconstruction
        '''

        unfiltered_us_rec, filtered_us_rec, filtered_fs_rec = batch
        
        modl_rec = self.model(unfiltered_us_rec)

        psnr_fbp_loss = self.loss_dict['psnr_loss'](filtered_us_rec, filtered_fs_rec)
        ssim_fbp_loss = self.loss_dict['ssim_loss'](filtered_us_rec, filtered_fs_rec)

        self.log("val/psnr_fbp", self.psnr(psnr_fbp_loss))
        self.log("val/ssim_fbp", ssim_fbp_loss)

        psnr_loss = self.loss_dict['psnr_loss'](modl_rec['dc'+str(self.model.K)], filtered_fs_rec)
        ssim_loss = self.loss_dict['ssim_loss'](modl_rec['dc'+str(self.model.K)], filtered_fs_rec)
        
        self.log("val/psnr", self.psnr(psnr_loss))
        self.log("val/ssim", ssim_loss)

        if self.loss_dict['loss_name'] == 'psnr':
            
            return psnr_loss
        
        elif self.loss_dict['loss_name'] == 'ssim':
            
            return ssim_loss

    def test_step(self, batch, batch_idx):

        '''
        Testing step for modl. 
        Suffixes:
            - 'us' stands for undersampled reconstruction (used as input with unfiltered backprojection)
            - 'fs' stands for fully sampled reconstruction
        '''

        unfiltered_us_rec, filtered_us_rec, filtered_fs_rec = batch
        
        psnr_fbp_loss = self.loss_dict['psnr_loss'](filtered_us_rec, filtered_fs_rec)
        ssim_fbp_loss = self.loss_dict['ssim_loss'](filtered_us_rec, filtered_fs_rec)

        self.log("test/psnr_fbp", self.psnr(psnr_fbp_loss))
        self.log("test/ssim_fbp", ssim_fbp_loss)

        modl_rec = self.model(unfiltered_us_rec)

        psnr_loss = self.loss_dict['psnr_loss'](modl_rec['dc'+str(self.model.K)], filtered_fs_rec)
        ssim_loss = self.loss_dict['ssim_loss'](modl_rec['dc'+str(self.model.K)], filtered_fs_rec)
        
        self.log("test/psnr", self.psnr(psnr_loss).item())
        self.log("test/ssim", ssim_loss.item())

        if self.loss_dict['loss_name'] == 'psnr':
            
            return psnr_loss
        
        elif self.loss_dict['loss_name'] == 'ssim':
            
            return ssim_loss
    
    def configure_optimizers(self):
        '''
        Configure optimizer
        '''
        if self.optimizer_dict['optimizer_name'] == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.optimizer_dict['lr'])
        return optimizer

    def process_kwdictionary(self, kw_dict):
        '''
        Process keyword dictionary.
        Params: 
            - kw_dictionary (dict): Dictionary with keywords
        '''
        
        self.optimizer_dict = kw_dict.pop('optimizer_dict')
        self.kw_dictionary_modl = kw_dict.pop('kw_dictionary_modl')
        self.loss_dict = kw_dict.pop('loss_dict')

        self.hparams['loss_dict'] = self.loss_dict
        self.hparams['kw_dictionary_modl'] = self.kw_dictionary_modl
        self.hparams['optimizer_dict'] = self.optimizer_dict
    
    @staticmethod
    def psnr(mse):
        '''
        Calculates PSNR respect to MSE mean value
        '''

        return 10*np.log10(1.0/mse.cpu().detach().numpy())



