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
        self.process_kwdictionary(kw_dictionary_model_system)

        self.model = modl.modl(self.kw_dictionary_modl)
    
    def forward(self, x):

        return self.model(x)
 
    def training_step(self, batch):
        '''
        Training step for modl. 
        Suffixes:
            - 'us' stands for undersampled reconstruction (used as input with unfiltered backprojection)
            - 'fs' stands for fully sampled reconstruction
        '''

        unfiltered_us_rec, filtered_us_rec, filtered_fs_rec = batch
        
        modl_rec = self.model(unfiltered_us_rec)
        loss = self.loss_dict['loss_name'](modl_rec, filtered_fs_rec)

        return loss

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



