'''
This code provides with an easy way to avoid boilerplate for training and validating results.
author: obanmarcos
'''
import sys
sys.path.append('~/DeepOPT/')

import torch
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
from skimage.transform import radon, iradon
from . import unet, modl
from . import alternating as altmodels
from collections.abc import Iterable
import wandb 
from timm.scheduler import TanhLRScheduler

# Modify for multi-gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

new_float_type = {
    # preserved types
    np.float32().dtype.char: np.float32,
    np.float64().dtype.char: np.float64,
    np.complex64().dtype.char: np.complex64,
    np.complex128().dtype.char: np.complex128,
    # altered types
    np.float16().dtype.char: np.float32,
    'g': np.float64,      # np.float128 ; doesn't exist on windows
    'G': np.complex128,   # np.complex256 ; doesn't exist on windows
}

dtype_range = {bool: (False, True),
               np.bool_: (False, True),
               np.bool8: (False, True),
               float: (-1, 1),
               np.float_: (-1, 1),
               np.float16: (-1, 1),
               np.float32: (-1, 1),
               np.float64: (-1, 1)}

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

        self.hR = lambda x: radon(x, angles, circle = False)
        self.hRT = lambda sino: iradon(sino, angles, circle = False)
        self.Psi = lambda x,th: altmodels.TVdenoise(x,
                                            2/th,
                                            self.tv_iters)
        self.Phi = lambda x: altmodels.TVnorm(x)

        self.model = modl.modl(self.kw_dictionary_modl)

        if self.track_alternating_admm == True:
            
            angles = np.linspace(0, 2*180, self.admm_dictionary['number_projections'], endpoint = False)
            
            self.ADMM = lambda y, y_true: altmodels.ADMM(y = self.hR(y), 
                                                        A = self.hR, 
                                                        AT = self.hRT, 
                                                        Den = self.Psi, 
                                                        alpha = self.admm_dictionary['alpha'],
                                                        delta = self.admm_dictionary['delta'], 
                                                        max_iter = self.admm_dictionary['max_iter'], 
                                                        phi = self.Phi, 
                                                        tol = self.admm_dictionary['tol'], 
                                                        invert = self.admm_dictionary['use_invert'], 
                                                        warm = self.admm_dictionary['use_warm_init'],
                                                        true_img = y_true,
                                                        verbose = self.admm_dictionary['verbose'])

        if self.track_alternating_twist == True:
            
            angles = np.linspace(0, 2*180, self.twist_dictionary['number_projections'], endpoint = False)
            
            kwarg = {'PSI': self.Psi, 'PHI': self.Phi, 'LAMBDA':self.twist_dictionary['lambda'], 'TOLERANCEA': self.twist_dictionary['tolerance'], 'STOPCRITERION': self.twist_dictionary['stop_criterion'], 'VERBOSE': self.twist_dictionary['verbose'], 'INITIALIZATION': self.twist_dictionary['initialization'], 'MAXITERA':self.twist_dictionary['max_iter'], 'GPU' : self.twist_dictionary['gpu']}
            
            self.TwIST = lambda y, y_true: altmodels.TwIST(y = self.hR(y), 
                                                        A = self.hR, 
                                                        AT = self.hRT, 
                                                        tau = self.twist_dictionary['tau'], 
                                                        kwarg = kwarg,true_img = y_true)

        if self.track_alternating_admm == True:
            
            angles = np.linspace(0, 2*180, self.admm_dictionary['number_projections'], endpoint = False)
            
            self.ADMM = lambda y, y_true: altmodels.ADMM(y = self.hR(y), 
                                                        A = self.hR, 
                                                        AT = self.hRT, 
                                                        Den = self.Psi, 
                                                        alpha = self.admm_dictionary['alpha'],
                                                        delta = self.admm_dictionary['delta'], 
                                                        max_iter = self.admm_dictionary['max_iter'], 
                                                        phi = self.Phi, 
                                                        tol = self.admm_dictionary['tol'], 
                                                        invert = self.admm_dictionary['use_invert'], 
                                                        warm = self.admm_dictionary['use_warm_init'],
                                                        true_img = y_true,
                                                        verbose = self.admm_dictionary['verbose'])

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

        unfiltered_us_rec = self.normalize_image_01(unfiltered_us_rec)
        filtered_us_rec = self.normalize_image_01(filtered_us_rec)
        filtered_fs_rec = self.normalize_image_01(filtered_fs_rec)

        if (self.track_train == True) and (batch_idx%500 == 0):

            self.log_plot(filtered_fs_rec, modl_rec, 'train')

        modl_rec = self.normalize_image_01(modl_rec['dc'+str(self.model.K)])
                
        psnr_fbp_loss = self.loss_dict['psnr_loss'](filtered_us_rec, filtered_fs_rec)
        ssim_fbp_loss = 1-self.loss_dict['ssim_loss'](filtered_us_rec, filtered_fs_rec)

        self.log("train/psnr_fbp", self.psnr(psnr_fbp_loss), on_step = True, on_epoch = False, prog_bar=True)
        self.log("train/ssim_fbp", 1-ssim_fbp_loss, on_step = True, on_epoch = False, prog_bar=True)

        psnr_loss = self.loss_dict['psnr_loss'](modl_rec, filtered_fs_rec)
        ssim_loss = 1-self.loss_dict['ssim_loss'](self.normalize_image_01(modl_rec), filtered_fs_rec)

        self.log("train/psnr", self.psnr(psnr_loss), on_step = True, on_epoch = False, prog_bar=True)
        self.log("train/ssim", 1-ssim_loss, on_step = True, on_epoch = False, prog_bar=True)

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

        unfiltered_us_rec = self.normalize_image_01(unfiltered_us_rec)
        filtered_us_rec = self.normalize_image_01(filtered_us_rec)
        filtered_fs_rec = self.normalize_image_01(filtered_fs_rec)

        if (self.track_val == True) and (batch_idx%500 == 0):

            self.log_plot(filtered_fs_rec, modl_rec, 'val')

        modl_rec = self.normalize_image_01(modl_rec['dc'+str(self.model.K)])
                
        psnr_fbp_loss = self.loss_dict['psnr_loss'](filtered_us_rec, filtered_fs_rec)
        ssim_fbp_loss = 1-self.loss_dict['ssim_loss'](filtered_us_rec, filtered_fs_rec)

        self.log("val/psnr_fbp", self.psnr(psnr_fbp_loss), on_step = True, on_epoch = False, prog_bar=True)
        self.log("val/ssim_fbp", 1-ssim_fbp_loss, on_step = True, on_epoch = False, prog_bar=True)

        psnr_loss = self.loss_dict['psnr_loss'](modl_rec, filtered_fs_rec)
        ssim_loss = 1-self.loss_dict['ssim_loss'](modl_rec, filtered_fs_rec)

        self.log("val/psnr", self.psnr(psnr_loss), on_step = True, on_epoch = False, prog_bar=True)
        self.log("val/ssim", 1-ssim_loss, on_step = True, on_epoch = False, prog_bar=True)

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

        if batch_idx != 30:

            return

        unfiltered_us_rec, filtered_us_rec, filtered_fs_rec = batch

        modl_rec = self.model(unfiltered_us_rec)
        
        ssim_loss = 1-self.loss_dict['ssim_loss'](self.normalize_image_01(modl_rec['dc'+str(self.model.K)]), self.normalize_image_01(filtered_fs_rec))

        unfiltered_us_rec = self.normalize_image_01(unfiltered_us_rec)
        unfiltered_us_rec = (unfiltered_us_rec-unfiltered_us_rec.mean())/unfiltered_us_rec.std()
        
        filtered_us_rec = self.normalize_image_01(filtered_us_rec)
        filtered_us_rec = (filtered_us_rec-filtered_us_rec.mean())/filtered_us_rec.std()
        
        filtered_fs_rec = self.normalize_image_01(filtered_fs_rec)
        filtered_fs_rec = (filtered_fs_rec-filtered_fs_rec.mean())/filtered_fs_rec.std()

        if (self.track_test == True) and (batch_idx == 0):

            self.log_plot(filtered_fs_rec, modl_rec, 'test')

        modl_rec = self.normalize_image_01(modl_rec['dc'+str(self.model.K)])
        modl_rec = (modl_rec - modl_rec.mean())/modl_rec.std()

        psnr_fbp_loss = self.loss_dict['psnr_loss'](filtered_us_rec, filtered_fs_rec)
        ssim_fbp_loss = 1-self.loss_dict['ssim_loss'](self.normalize_image_01(filtered_us_rec), self.normalize_image_01(filtered_fs_rec))

        self.log("test/psnr_fbp", self.psnr(filtered_us_rec, filtered_fs_rec), on_step = True, prog_bar = True)
        self.log("test/ssim_fbp", (1-ssim_fbp_loss).cpu(), on_step = True, prog_bar = True)

        psnr_loss = self.loss_dict['psnr_loss'](modl_rec, filtered_fs_rec)

        self.log("test/psnr", self.psnr(modl_rec, filtered_fs_rec), on_step = True,prog_bar = True)
        self.log("test/ssim", (1-ssim_loss).cpu(), on_step = True, prog_bar = True)

        self.test_metric['test/psnr'].append(self.psnr(modl_rec, filtered_fs_rec))
        self.test_metric['test/ssim'].append((1-ssim_loss).cpu())
        
        self.test_metric['test/psnr_fbp'].append(self.psnr(filtered_us_rec, filtered_fs_rec))
        self.test_metric['test/ssim_fbp'].append((1-ssim_fbp_loss).cpu())
        
        if self.track_alternating_admm == True:

            admm_rec = torch.zeros_like(filtered_fs_rec)

            for i, (filt_us_rec_image, filt_fs_rec_image) in enumerate(zip(filtered_us_rec, filtered_fs_rec)):
                
                admm_rec[i,0, ...] = torch.FloatTensor(self.ADMM(filt_us_rec_image.cpu().numpy()[0,...].T, filt_fs_rec_image.cpu().numpy()[0,...].T)[0])

            admm_rec = (admm_rec - admm_rec.mean())/admm_rec.std()
            
            loss_psnr_admm = self.psnr(admm_rec, filtered_fs_rec)
            loss_ssim_admm = 1-self.loss_dict['ssim_loss'](self.normalize_image_01(admm_rec), self.normalize_image_01(filtered_fs_rec))

            self.log("test/psnr_admm", loss_psnr_admm, on_step = True,prog_bar = True)
            self.log("test/ssim_admm", (1-loss_ssim_admm).cpu(), on_step = True, prog_bar = True)
            
            self.test_metric['test/psnr_admm'].append(loss_psnr_admm)
            self.test_metric['test/ssim_admm'].append((1-loss_ssim_admm).cpu())

        if self.track_alternating_twist == True:

            twist_rec = torch.zeros_like(filtered_fs_rec)

            for i, (filt_us_rec_image, filt_fs_rec_image) in enumerate(zip(filtered_us_rec, filtered_fs_rec)):
                
                twist_rec[i,0, ...] = torch.FloatTensor(self.TwIST(filt_us_rec_image.cpu().numpy()[0,...].T, filt_fs_rec_image.cpu().numpy()[0,...].T)[0])

            twist_rec = self.normalize_image_01(twist_rec)
            twist_rec = (twist_rec - twist_rec.mean())/twist_rec.std()
        
            loss_psnr_twist = self.psnr(twist_rec, filtered_fs_rec)
            loss_ssim_twist = 1-self.loss_dict['ssim_loss'](self.normalize_image_01(twist_rec), self.normalize_image_01(filtered_fs_rec))

            self.log("test/psnr_twist", loss_psnr_twist, on_step = True,prog_bar = True)
            self.log("test/ssim_twist", (1-loss_ssim_twist).cpu(), on_step = True, prog_bar = True)
            
            self.test_metric['test/psnr_twist'].append(loss_psnr_twist)
            self.test_metric['test/ssim_twist'].append((1-loss_ssim_twist).cpu())

        if self.loss_dict['loss_name'] == 'psnr':

            return psnr_loss
        
        elif self.loss_dict['loss_name'] == 'ssim':
        
            return ssim_loss

    def create_test_metric(self):

        self.test_metric = {'test/psnr_twist': [], 'test/ssim_twist': [], 'test/psnr_admm': [], 'test/ssim_admm': [],'test/psnr':[], 'test/ssim':[], 'test/psnr_fbp':[], 'test/ssim_fbp':[]}

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

        self.track_alternating_admm = kw_dict['track_alternating_admm']
        self.track_alternating_twist = kw_dict['track_alternating_twist']

        if self.track_alternating_admm == True:

            self.admm_dictionary = kw_dict['admm_dictionary']
        
        if self.track_alternating_twist == True:

            self.twist_dictionary = kw_dict['twist_dictionary']

        self.tv_iters = kw_dict['tv_iters']
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
    def psnr(imgs1, imgs2):

        psnr_list = []

        for img1, img2 in zip(imgs1, imgs2):
            
            img_range = img1[0,...].max() - img1[0,...].min()
            
            psnr_list.append(_psnr(img1[0,...].detach().cpu().numpy(), img2[0,...].detach().cpu().numpy(), data_range = img_range.cpu().numpy())) 
        
        return np.array(psnr_list).mean()
    
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
        plt.close(fig)

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

def _as_floats(image0, image1):
    """
    Promote im1, im2 to nearest appropriate floating point precision.
    """
    float_type = _supported_float_type([image0.dtype, image1.dtype])
    image0 = np.asarray(image0, dtype=float_type)
    image1 = np.asarray(image1, dtype=float_type)
    
    return image0, image1

def _supported_float_type(input_dtype, allow_complex=False):
    """Return an appropriate floating-point dtype for a given dtype.
    float32, float64, complex64, complex128 are preserved.
    float16 is promoted to float32.
    complex256 is demoted to complex128.
    Other types are cast to float64.
    Parameters
    ----------
    input_dtype : np.dtype or Iterable of np.dtype
        The input dtype. If a sequence of multiple dtypes is provided, each
        dtype is first converted to a supported floating point type and the
        final dtype is then determined by applying `np.result_type` on the
        sequence of supported floating point types.
    allow_complex : bool, optional
        If False, raise a ValueError on complex-valued inputs.
    Returns
    -------
    float_type : dtype
        Floating-point dtype for the image.
    """
    if isinstance(input_dtype, Iterable) and not isinstance(input_dtype, str):
        return np.result_type(*(_supported_float_type(d) for d in input_dtype))
    input_dtype = np.dtype(input_dtype)
    if not allow_complex and input_dtype.kind == 'c':
        raise ValueError("complex valued input is not supported")
    return new_float_type.get(input_dtype.char, np.float64)

def mean_squared_error(image0, image1):
    """
    Compute the mean-squared error between two images.
    Parameters
    ----------
    image0, image1 : ndarray
        Images.  Any dimensionality, must have same shape.
    Returns
    -------
    mse : float
        The mean-squared error (MSE) metric.
    Notes
    -----
    .. versionchanged:: 0.16
        This function was renamed from ``skimage.measure.compare_mse`` to
        ``skimage.metrics.mean_squared_error``.
    """
    image0, image1 = _as_floats(image0, image1)
    return np.mean((image0 - image1) ** 2, dtype=np.float64)

def _psnr(image_test, image_true, data_range = None):
    '''
    Calculates PSNR respect to MSE mean value
    '''

    if data_range is None:
        if image_true.dtype != image_test.dtype:
            warn("Inputs have mismatched dtype.  Setting data_range based on "
                "image_true.")
        dmin, dmax = dtype_range[image_true.dtype.type]
        true_min, true_max = np.min(image_true), np.max(image_true)
        if true_max > dmax or true_min < dmin:
            raise ValueError(
                "image_true has intensity values outside the range expected "
                "for its data type. Please manually specify the data_range.")
        if true_min >= 0:
            # most common case (255 for uint8, 1 for float)
            data_range = dmax
        else:
            data_range = dmax - dmin

    image_true, image_test = _as_floats(image_true, image_test)

    err = mean_squared_error(image_true, image_test)
    
    return 10 * np.log10((data_range ** 2) / err) 

class UNetReconstructor(pl.LightningModule):
    '''
    Pytorch Lightning for U-Net boilerplate
    '''
    def __init__(self, kw_dictionary_model_system):
        '''
        Initializes U-Net reconstructor. 
        Params:
            - kw_dictionary_model_system (dict): 
        '''
        super().__init__()
        
        # wandb.init(project = 'deepopt')

        self.process_kwdictionary(kw_dictionary_model_system)

        self.model = unet.unet(self.kw_dictionary_unet)

        self.save_hyperparameters(self.hparams)

    def forward(self, x):

        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        '''
        Training step for unet. 
        Suffixes:
            - 'us' stands for undersampled reconstruction (used as input with unfiltered backprojection)
            - 'fs' stands for fully sampled reconstruction
        '''

        unfiltered_us_rec, filtered_us_rec, filtered_fs_rec = batch

        unet_rec = self.model(filtered_us_rec)

        if (self.track_train == True) and (batch_idx%500 == 0):

            self.log_plot(filtered_fs_rec, unet_rec, filtered_fs_rec, 'train')
                
        psnr_fbp_loss = self.loss_dict['psnr_loss'](filtered_us_rec, filtered_fs_rec)
        ssim_fbp_loss = 1-self.loss_dict['ssim_loss'](self.normalize_image_01(filtered_us_rec), self.normalize_image_01(filtered_fs_rec))

        self.log("train/psnr_fbp", self.psnr(psnr_fbp_loss, range_max_min = [filtered_fs_rec.max(), filtered_fs_rec.min()]), on_step = True, on_epoch = False, prog_bar=True)
        self.log("train/ssim_fbp", 1-ssim_fbp_loss, on_step = True, on_epoch = False, prog_bar=True)

        psnr_loss = self.loss_dict['psnr_loss'](unet_rec, filtered_fs_rec)
        ssim_loss = 1-self.loss_dict['ssim_loss'](self.normalize_image_01(unet_rec), self.normalize_image_01(filtered_fs_rec))

        self.log("train/psnr", self.psnr(psnr_loss, range_max_min = [filtered_fs_rec.max(), filtered_fs_rec.min()]), on_step = True, on_epoch = False, prog_bar=True)
        self.log("train/ssim", 1-ssim_loss, on_step = True, on_epoch = False, prog_bar=True)


        if self.loss_dict['loss_name'] == 'psnr':
            
            if torch.isnan(psnr_loss):
                print('nan found, logging image')
                self.log_plot(filtered_fs_rec, unet_rec, 'train')

            return psnr_loss
        
        elif self.loss_dict['loss_name'] == 'ssim':
            
            if torch.isnan(ssim_loss):
                print('nan found, logging image')
                self.log_plot(filtered_fs_rec, unet_rec, 'train')

            return ssim_loss
            
        elif self.loss_dict['loss_name'] == 'mssim':

            msssim_loss = 1-self.loss_dict['msssim_loss'](unet_rec, filtered_fs_rec)
            self.log("train/msssim", msssim_loss)
            
            return msssim_loss
    
    def validation_step(self, batch, batch_idx):

        '''
        Validation step for unet. 
        Suffixes:
            - 'us' stands for undersampled reconstruction (used as input with unfiltered backprojection)
            - 'fs' stands for fully sampled reconstruction
        '''

        unfiltered_us_rec, filtered_us_rec, filtered_fs_rec = batch
        
        unet_rec = self.model(filtered_us_rec)

        if (self.track_val == True) and ((self.current_epoch == 0) or (self.current_epoch == self.max_epochs-1)) and (batch_idx == 0):

            self.log_plot(filtered_fs_rec, unet_rec, filtered_fs_rec, 'validation')

        psnr_fbp_loss = self.loss_dict['psnr_loss'](filtered_us_rec, filtered_fs_rec)
        ssim_fbp_loss = 1-self.loss_dict['ssim_loss'](self.normalize_image_01(filtered_us_rec), self.normalize_image_01(filtered_fs_rec))

        self.log("val/psnr_fbp", self.psnr(psnr_fbp_loss, range_max_min = [filtered_fs_rec.max(), filtered_fs_rec.min()]))
        self.log("val/ssim_fbp", 1-ssim_fbp_loss)

        psnr_loss = self.loss_dict['psnr_loss'](unet_rec, filtered_fs_rec)
        ssim_loss = 1-self.loss_dict['ssim_loss'](self.normalize_image_01(unet_rec), self.normalize_image_01(filtered_fs_rec))
        
        self.log("val/psnr", self.psnr(psnr_loss, range_max_min = [filtered_fs_rec.max(), filtered_fs_rec.min()]))
        self.log("val/ssim", 1-ssim_loss)

        if self.loss_dict['loss_name'] == 'psnr':
            
            return psnr_loss
        
        elif self.loss_dict['loss_name'] == 'ssim':
            
            return ssim_loss
        
        elif self.loss_dict['loss_name'] == 'msssim':
            
            msssim_loss = 1-self.loss_dict['msssim_loss'](unet_rec, filtered_fs_rec)
            self.log("val/msssim", msssim_loss)

            return msssim_loss

    def test_step(self, batch, batch_idx):

        '''
        Testing step for unet. 
        Suffixes:
            - 'us' stands for undersampled reconstruction (used as input with unfiltered backprojection)
            - 'fs' stands for fully sampled reconstruction
        '''

        unfiltered_us_rec, filtered_us_rec, filtered_fs_rec = batch
        
        psnr_fbp_loss = self.loss_dict['psnr_loss'](filtered_us_rec, filtered_fs_rec)
        ssim_fbp_loss = 1-self.loss_dict['ssim_loss'](self.normalize_image_01(filtered_us_rec), self.normalize_image_01(filtered_fs_rec))

        self.log("test/psnr_fbp", self.psnr(psnr_fbp_loss, range_max_min = [filtered_fs_rec.max(), filtered_fs_rec.min()]))
        self.log("test/ssim_fbp", 1-ssim_fbp_loss)

        unet_rec = self.model(filtered_us_rec)

        if (self.track_test == True) and (batch_idx == 0):

            self.log_plot(filtered_fs_rec, unet_rec, filtered_fs_rec, 'test')

        psnr_loss = self.loss_dict['psnr_loss'](unet_rec, filtered_fs_rec)
        ssim_loss = 1-self.loss_dict['ssim_loss'](self.normalize_image_01(unet_rec), self.normalize_image_01(filtered_fs_rec))
        
        self.log("test/psnr", self.psnr(psnr_loss, range_max_min = [filtered_fs_rec.max(), filtered_fs_rec.min()]).item())
        self.log("test/ssim", 1-ssim_loss.item())

        if self.loss_dict['loss_name'] == 'psnr':
            
            return psnr_loss
        
        elif self.loss_dict['loss_name'] == 'ssim':
            
            return ssim_loss
        
        elif self.loss_dict['loss_name'] == 'msssim':
            
            msssim_loss = 1-self.loss_dict['msssim_loss'](unet_rec['dc'+str(self.model.K)], filtered_fs_rec)
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
        self.kw_dictionary_unet = kw_dict['kw_dictionary_unet']
        self.loss_dict = kw_dict['loss_dict']
        self.max_epochs = kw_dict['max_epochs']

        self.track_train = kw_dict['track_train']
        self.track_val = kw_dict['track_val']
        self.track_test = kw_dict['track_test']

        self.hparams['loss_dict'] = self.loss_dict
        self.hparams['kw_dictionary_unet'] = self.kw_dictionary_unet
        self.hparams['optimizer_dict'] = self.optimizer_dict

    @staticmethod
    def psnr(mse, range_max_min = [0,1]):
        '''
        Calculates PSNR respect to MSE mean value
        '''

        return 10*torch.log10((range_max_min[1]-range_max_min[0])**2/mse)

    def log_plot(self, target, prediction, benchmark, phase):
        '''
        Plots target and prediction (unrolled) and logs it. 
        '''
        
        fig, ax = plt.subplots(1, 3, figsize = (16,6))
        
        im = ax[0].imshow(target.detach().cpu().numpy()[0,0,:,:], cmap = 'gray')
        ax[0].set_title('Target')
        ax[0].axis('off') 
        
        plt.suptitle('Epoch {} in {} phase'.format(self.current_epoch, phase))

        im = ax[1].imshow(prediction.detach().cpu().numpy()[0,0,:,:], cmap = 'gray')
        ax[1].set_title('U-Net reconstruction')
        ax[1].axis('off')

        im = ax[2].imshow(benchmark.detach().cpu().numpy()[0,0,:,:], cmap = 'gray')
        ax[2].set_title('FBP reconstruction')
        ax[2].axis('off')
        
        cax = fig.add_axes([ax[2].get_position().x1+0.01,ax[2].get_position().y0,0.02, ax[2].get_position().height])
        plt.colorbar(im, cax = cax)

        wandb.log({'epoch':self.current_epoch, '{}_plot_{}'.format(phase, self.current_epoch): fig})
        
        plt.close(fig)

    def log_samples(self, batch, model_reconstruction):
        '''
        Logs images from training.
        '''

        unfiltered_us_rec, filtered_us_rec, filtered_fs_rec = batch

        image_tensor = [unfiltered_us_rec[0,...], filtered_us_rec[0,...], filtered_fs_rec[0,...], model_reconstruction[0, ...]]

        image_grid = torchvision.utils.make_grid(image_tensor)
        image_grid = wandb.Image(image_grid, caption="Left: Unfiltered undersampled backprojection\n Center 1 : Filtered undersampled backprojection\nCenter 2: Filtered fully sampled\n Right: unet reconstruction")

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
