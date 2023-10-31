from pytorch_msssim import SSIM
from torchmetrics import MultiScaleStructuralSimilarityIndexMeasure as MSSSIM
import torch

acceleration_factor = 20

# ++++++++++++++++++++++++++++++++++
#            MODEL
# ++++++++++++++++++++++++++++++++++
# ResNet dictionary parameters
resnet_options_dict = {'number_layers': 8,
                        'kernel_size':3,
                        'features':64,
                        'in_channels':1,
                        'out_channels':1,
                        'stride':1, 
                        'use_batch_norm': True,
                        'init_method': 'xavier'}

# Model parameters
modl_dict = {'use_torch_radon': False,
            'number_layers': 8,
            'K_iterations' : 8,
            'number_projections_total' : 720,
            'number_projections_undersampled' : 72, 
            'acceleration_factor':10,
            'image_size': 100,
            'lambda': 0.05,
            'use_shared_weights': True,
            'denoiser_method': 'resnet',
            'resnet_options': resnet_options_dict,
            'in_channels': 1,
            'out_channels': 1}

# U-Net parameters
unet_dict = {'n_channels': 1,
            'n_classes':1,
            'bilinear': True,
            'batch_norm': True,
            'batch_norm_inconv':True,
            'residual': False,
            'up_conv': False}

# TwIST parameters
twist_dictionary = {'number_projections': modl_dict['number_projections_total'], 
        'lambda': 1e-2, 
        'tolerance':1e-4,
        'stop_criterion':1, 
        'verbose':0,
        'initialization':0,
        'max_iter':10, 
        'gpu':0,
        'tau': 0.01}

# Training parameters
loss_dict = {'loss_name': 'psnr',
            'psnr_loss': torch.nn.MSELoss(reduction = 'mean'),
            'ssim_loss': SSIM(data_range=1, size_average=True, channel=1),
            'msssim_loss': MSSSIM(kernel_size = 1),
            'l1_loss' : torch.nn.L1Loss(reduction = 'mean')}

 

# Optimizer parameters
optimizer_dict = {'optimizer_name': 'Adam+Tanh',
            'lr': 1e-4}

# System parameters
model_system_dict = {'acc_factor_data': 20,
                        'use_normalize': True,
                        'optimizer_dict': optimizer_dict,
                        'kw_dictionary_modl': modl_dict,
                        'kw_dictionary_unet': unet_dict,
                        'loss_dict': loss_dict, 
                        'method':'modl',                       
                        'track_train': True,
                        'track_val': True,
                        'track_test': True,
                        'max_epochs': 25, 
                        'save_model':True,
                        'load_model': False,
                        'load_path': '',
                        'save_path': 'MoDL_K_fold_{}',
                        'track_alternating_admm':False,
                        'tv_iters': 20,
                        'title': 'HyperParams_Search',
                        'metrics_folder': '/path/to/metrics/',
                        'models_folder': '/path/to/metrics/',         
                        'track_alternating_twist': True,
                        'track_unet': False,
                        'twist_dictionary':twist_dictionary}

# +++++++++++++++++++++++++++++++++++
#            TRAINER
# +++++++++++++++++++++++++++++++++++

# PL Trainer and W&B logger dictionaries

model_folder = '/path/to/model'      
logger_dict = {'project':'tomodl',
                'entity': 'user', 
                'log_model': True}

lightning_trainer_dict = {'max_epochs': 40,
                            'log_every_n_steps': 10,
                            'check_val_every_n_epoch': 1,
                            'gradient_clip_val' : 0.5,
                            'accelerator' : 'gpu', 
                            'devices' : 1,
                            'fast_dev_run' : False,
                            'default_root_dir': model_folder}

profiler = None
# profiler = SimpleProfiler(dirpath = './logs/', filename = 'Test_training_profile_pytorch')
# profiler = PyTorchProfiler(dirpath = './logs/', filename = 'Test_training_profile_pytorch')

trainer_system_dict = {'lightning_trainer_dict': lightning_trainer_dict,
                'use_k_folding': True, 
                'track_checkpoints': True,
                'epoch_number_checkpoint': 10,
                'use_swa' : False,
                'use_accumulate_batches': False,
                'k_fold_number_datasets': 3,
                'use_logger' : True,
                'logger_dict': logger_dict,
                'track_default_checkpoints'  : True,
                'use_auto_lr_find': False,
                'batch_accumulate_number': 3,
                'use_mixed_precision': False,
                'batch_accumulation_start_epoch': 0, 
                'profiler': profiler, 
                'restore_fold': False,
                'resume': False}


# +++++++++++++++++++++++++++++++++++
#            DATALOADER
# +++++++++++++++++++++++++++++++++++

datasets_folder = '/path/to/datasets'
data_transform = None                                    

dataloader_system_dict = {'datasets_folder': datasets_folder,
                        'number_volumes' : 0,
                        'experiment_name': 'Bassi',
                        'img_resize': 100,
                        'load_shifts': True,
                        'save_shifts':False,
                        'number_projections_total': 720,
                        'number_projections_undersampled': 72,
                        'acceleration_factor':acceleration_factor,
                        'train_factor' : 0.8, 
                        'val_factor' : 0.2,
                        'test_factor' : 0.2, 
                        'batch_size' : 8, 
                        'sampling_method' : 'equispaced-linear',
                        'shuffle_data' : True,
                        'data_transform' : data_transform,
                        'num_workers':0,
                        'use_number_samples':False,
                        'number_samples_factor':1.0,
                        'use_subset_by_part': False}
