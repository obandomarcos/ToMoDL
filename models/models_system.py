'''
This code provides with an easy way to avoid boilerplate for training and validating results.
author: obanmarcos
'''
import sys
sys.path.append('~/DeepOPT/')

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
import torchvision
from . import unet
import wandb 
from timm.scheduler import TanhLRScheduler

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
        
        # wandb.init(project = 'deepopt')

        self.process_kwdictionary(kw_dictionary_model_system)

        self.model = modl.modl(self.kw_dictionary_modl)

        self.save_hyperparameters(self.hparams)

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

        if (self.track_train == True) and (batch_idx%50 == 0):

            self.log_plot(filtered_fs_rec, modl_rec, 'train')
                
        psnr_fbp_loss = self.loss_dict['psnr_loss'](filtered_us_rec, filtered_fs_rec)
        ssim_fbp_loss = 1-self.loss_dict['ssim_loss'](self.normalize_image_01(filtered_us_rec), self.normalize_image_01(filtered_fs_rec))

        self.log("train/psnr_fbp", self.psnr(psnr_fbp_loss, range_max_min = [filtered_fs_rec.max(), filtered_fs_rec.min()]), on_step = True, on_epoch = False, prog_bar=True)
        self.log("train/ssim_fbp", 1-ssim_fbp_loss, on_step = True, on_epoch = False, prog_bar=True)

        psnr_loss = self.loss_dict['psnr_loss'](modl_rec['dc'+str(self.model.K)], filtered_fs_rec)
        ssim_loss = 1-self.loss_dict['ssim_loss'](self.normalize_image_01(modl_rec['dc'+str(self.model.K)]), self.normalize_image_01(filtered_fs_rec))

        self.log("train/psnr", self.psnr(psnr_loss, range_max_min = [filtered_fs_rec.max(), filtered_fs_rec.min()]), on_step = True, on_epoch = False, prog_bar=True)
        self.log("train/ssim", 1-ssim_loss, on_step = True, on_epoch = False, prog_bar=True)


        if self.loss_dict['loss_name'] == 'psnr':
            
            if torch.isnan(psnr_loss):
                print('nan found, logging image')
                self.log_plot(filtered_fs_rec, modl_rec, 'train')

            return psnr_loss
        
        elif self.loss_dict['loss_name'] == 'ssim':
            
            if torch.isnan(ssim_loss):
                print('nan found, logging image')
                self.log_plot(filtered_fs_rec, modl_rec, 'train')

            return ssim_loss
            
        elif self.loss_dict['loss_name'] == 'mssim':

            msssim_loss = 1-self.loss_dict['msssim_loss'](modl_rec['dc'+str(self.model.K)], filtered_fs_rec)
            self.log("train/msssim", msssim_loss)
            
            return msssim_loss
    
    def validation_step(self, batch, batch_idx):

        '''
        Validation step for modl. 
        Suffixes:
            - 'us' stands for undersampled reconstruction (used as input with unfiltered backprojection)
            - 'fs' stands for fully sampled reconstruction
        '''

        unfiltered_us_rec, filtered_us_rec, filtered_fs_rec = batch
        
        modl_rec = self.model(unfiltered_us_rec)

        if (self.track_val == True) and ((self.current_epoch == 0) or (self.current_epoch == self.max_epochs-1)) and (batch_idx == 0):

            self.log_plot(filtered_fs_rec, modl_rec, 'validation')

        psnr_fbp_loss = self.loss_dict['psnr_loss'](filtered_us_rec, filtered_fs_rec)
        ssim_fbp_loss = 1-self.loss_dict['ssim_loss'](self.normalize_image_01(filtered_us_rec), self.normalize_image_01(filtered_fs_rec))

        self.log("val/psnr_fbp", self.psnr(psnr_fbp_loss, range_max_min = [filtered_fs_rec.max(), filtered_fs_rec.min()]))
        self.log("val/ssim_fbp", 1-ssim_fbp_loss)

        psnr_loss = self.loss_dict['psnr_loss'](modl_rec['dc'+str(self.model.K)], filtered_fs_rec)
        ssim_loss = 1-self.loss_dict['ssim_loss'](self.normalize_image_01(modl_rec['dc'+str(self.model.K)]), self.normalize_image_01(filtered_fs_rec))
        
        self.log("val/psnr", self.psnr(psnr_loss, range_max_min = [filtered_fs_rec.max(), filtered_fs_rec.min()]))
        self.log("val/ssim", 1-ssim_loss)

        if self.loss_dict['loss_name'] == 'psnr':
            
            return psnr_loss
        
        elif self.loss_dict['loss_name'] == 'ssim':
            
            return ssim_loss
        
        elif self.loss_dict['loss_name'] == 'msssim':
            
            msssim_loss = 1-self.loss_dict['msssim_loss'](modl_rec['dc'+str(self.model.K)], filtered_fs_rec)
            self.log("val/msssim", msssim_loss)

            return msssim_loss

    def test_step(self, batch, batch_idx):

        '''
        Testing step for modl. 
        Suffixes:
            - 'us' stands for undersampled reconstruction (used as input with unfiltered backprojection)
            - 'fs' stands for fully sampled reconstruction
        '''

        unfiltered_us_rec, filtered_us_rec, filtered_fs_rec = batch
        
        psnr_fbp_loss = self.loss_dict['psnr_loss'](filtered_us_rec, filtered_fs_rec)
        ssim_fbp_loss = 1-self.loss_dict['ssim_loss'](self.normalize_image_01(filtered_us_rec), self.normalize_image_01(filtered_fs_rec))

        self.log("test/psnr_fbp", self.psnr(psnr_fbp_loss, range_max_min = [filtered_fs_rec.max(), filtered_fs_rec.min()]))
        self.log("test/ssim_fbp", 1-ssim_fbp_loss)

        modl_rec = self.model(unfiltered_us_rec)

        if (self.track_test == True) and (batch_idx == 0):

            self.log_plot(filtered_fs_rec, modl_rec, 'test')

        psnr_loss = self.loss_dict['psnr_loss'](modl_rec['dc'+str(self.model.K)], filtered_fs_rec)
        ssim_loss = 1-self.loss_dict['ssim_loss'](self.normalize_image_01(modl_rec['dc'+str(self.model.K)]), self.normalize_image_01(filtered_fs_rec))
        
        self.log("test/psnr", self.psnr(psnr_loss, range_max_min = [filtered_fs_rec.max(), filtered_fs_rec.min()]).item(), on_step = True, on_epoch = False)
        self.log("test/ssim", 1-ssim_loss.item(), on_step = True, on_epoch = False)

        if self.loss_dict['loss_name'] == 'psnr':
            
            return psnr_loss
        
        elif self.loss_dict['loss_name'] == 'ssim':
            
            return ssim_loss
        
        elif self.loss_dict['loss_name'] == 'msssim':
            
            msssim_loss = 1-self.loss_dict['msssim_loss'](modl_rec['dc'+str(self.model.K)], filtered_fs_rec)
            self.log("test/msssim", msssim_loss)

            return msssim_loss

    
    def configure_optimizers(self):
        '''
        Configure optimizer
        '''
        if self.optimizer_dict['optimizer_name'] == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(),
            lr=self.optimizer_dict['lr'])
            
            return optimizer
        
        if self.optimizer_dict['optimizer_name'] == 'Adam+Tanh':
            
            optimizer = torch.optim.Adam(self.parameters(),
            lr=self.optimizer_dict['lr'])
            scheduler = TanhLRScheduler(optimizer, self.max_epochs-1)
            
            return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
            
    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        '''
        Lr scheduler step
        '''
        scheduler.step(epoch=self.current_epoch)
    def process_kwdictionary(self, kw_dict):
        '''
        Process keyword dictionary.
        Params: 
            - kw_dictionary (dict): Dictionary with keywords
        '''
        
        self.optimizer_dict = kw_dict['optimizer_dict']
        self.kw_dictionary_modl = kw_dict['kw_dictionary_modl']
        self.loss_dict = kw_dict['loss_dict']
        self.max_epochs = kw_dict['max_epochs']

        self.track_train = kw_dict['track_train']
        self.track_val = kw_dict['track_val']
        self.track_test = kw_dict['track_test']

        self.hparams['loss_dict'] = self.loss_dict
        self.hparams['kw_dictionary_modl'] = self.kw_dictionary_modl
        self.hparams['optimizer_dict'] = self.optimizer_dict

    @staticmethod
    def psnr(mse, range_max_min = [0,1]):
        '''
        Calculates PSNR respect to MSE mean value
        '''

        return 10*torch.log10((range_max_min[1]-range_max_min[0])**2/mse)

    def log_plot(self, target, prediction, phase):
        '''
        Plots target and prediction (unrolled) and logs it. 
        '''
        
        fig, ax = plt.subplots(1, len(prediction.keys())+1, figsize = (16,6))
        
        im = ax[0].imshow(target.detach().cpu().numpy()[0,0,:,:], cmap = 'gray')
        ax[0].set_title('Target')
        ax[0].axis('off') 
        
        plt.suptitle('Epoch {} in {} phase'.format(self.current_epoch, phase))

        for a, (key, image) in zip(ax[1:], prediction.items()):

            im = a.imshow(image.detach().cpu().numpy()[0,0,:,:], cmap = 'gray')
            a.set_title(key)
            a.axis('off')
        
        cax = fig.add_axes([a.get_position().x1+0.01,a.get_position().y0,0.02,a.get_position().height])
        plt.colorbar(im, cax = cax)

        wandb.log({'{}_plot_{}'.format(phase, self.current_epoch): fig})

    def log_samples(self, batch, model_reconstruction):
        '''
        Logs images from training.
        '''

        unfiltered_us_rec, filtered_us_rec, filtered_fs_rec = batch

        image_tensor = [unfiltered_us_rec[0,...], filtered_us_rec[0,...], filtered_fs_rec[0,...], model_reconstruction[0, ...]]

        image_grid = torchvision.utils.make_grid(image_tensor)
        image_grid = wandb.Image(image_grid, caption="Left: Unfiltered undersampled backprojection\n Center 1 : Filtered undersampled backprojection\nCenter 2: Filtered fully sampled\n Right: MoDL reconstruction")

        wandb.log({'images {}'.format(self.current_epoch): image_grid})

    def load_model(self):
        '''
        TO-DO: 
        * Add method for model loading from checkpoint
            * Load names from versions and choose best k.
        '''
        pass
    
    @staticmethod
    def normalize_image_01(images):
        '''
        Normalizes tensor of images 1-channel images between 0 and 1.
        Params:
        - images (torch.Tensor): Tensor of 1-channel images
        '''
        
        image_norm = torch.zeros_like(images)

        for i, image in enumerate(images):
            
            image_norm[i,...] = ((image - image.min())/(image.max()-image.min()))

        return image_norm        
