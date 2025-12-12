'''
K-Folding script
author: obanmarcos
'''
import os
import os, sys
from config import * 
import wandb

sys.path.append(where_am_i())

import lightning as pl
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utilities import dataloading_utilities as dlutils
from utilities.folders import *

from training import train_utilities as trutils

from models.models_system import MoDLReconstructor, UNetReconstructor
import torch


from lightning.callbacks import ModelCheckpoint
from lightning.loggers import WandbLogger

from torchvision import transforms as T
from pytorch_msssim import SSIM
# from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure as MSSSIM

# Options for folding menu
use_default_model_dict = True
use_default_dataloader_dict = True
use_default_trainer_dict = True

profiler = None
model_dict = {'psnr':
                {'2': 
                {'models':
                    {'0':'omarcos/deepopt/model-9r89t9j2:v0',
                     '1':'omarcos/deepopt/model-1nlkmche:v0',
                     '2':'omarcos/deepopt/model-lyt89k5t:v0',
                     '3':'omarcos/deepopt/model-2s01fb36:v0'
                    },
                    'order_0' : ['/home/obanmarcos/Balseiro/DeepOPT/datasets/x2/140519_5dpf_head_2', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x2/140117_3dpf_lower tail_2', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x2/140714_5dpf_head_2', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x2/140315_3dpf_head_2', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x2/140117_3dpf_head_2', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x2/140114_5dpf_head_2', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x2/140114_5dpf_upper tail_2', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x2/140117_3dpf_body_2', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x2/140315_1dpf_head_2', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x2/140117_3dpf_upper tail_2', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x2/140114_5dpf_body_2', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x2/140114_5dpf_lower tail_2']
                },
                '6':
                {'models':
                {'0':'omarcos/deepopt/model-bu7hlp3b:v0',
                     '1':'omarcos/deepopt/model-5sd2dysq:v0',
                     '2':'omarcos/deepopt/model-d58tvubf:v0',
                     '3':'omarcos/deepopt/model-1erq431o:v0'
                    },
                'order_0': ['/home/obanmarcos/Balseiro/DeepOPT/datasets/x6/140117_3dpf_lower tail_6', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x6/140714_5dpf_head_6', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x6/140114_5dpf_lower tail_6', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x6/140114_5dpf_upper tail_6', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x6/140117_3dpf_body_6', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x6/140519_5dpf_head_6', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x6/140117_3dpf_head_6', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x6/140315_3dpf_head_6', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x6/140114_5dpf_body_6','/home/obanmarcos/Balseiro/DeepOPT/datasets/x6/140315_1dpf_head_6', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x6/140117_3dpf_upper tail_6', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x6/140114_5dpf_head_6']
                },
                '10': 
                {'models':
                {'0':'omarcos/deepopt/model-203eopxg:v0',
                     '1':'omarcos/deepopt/model-oyxvdcj7:v0',
                     '2':'omarcos/deepopt/model-3fmp7pqn:v0',
                     '3':'omarcos/deepopt/model-zx2zo5x1:v0'},
                'order_0': ['/home/obanmarcos/Balseiro/DeepOPT/datasets/x10/140114_5dpf_upper tail_10', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x10/140315_3dpf_head_10', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x10/140117_3dpf_body_10', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x10/140315_1dpf_head_10', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x10/140117_3dpf_upper tail_10', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x10/140519_5dpf_head_10', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x10/140117_3dpf_head_10', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x10/140117_3dpf_lower tail_10', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x10/140114_5dpf_head_10', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x10/140714_5dpf_head_10', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x10/140114_5dpf_lower tail_10', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x10/140114_5dpf_body_10']
                },
                '14':
                {'models':
                {'0':'omarcos/deepopt/model-t96gkwcs:v0',
                     '1':'omarcos/deepopt/model-1jkmmgtf:v0',
                     '2':'omarcos/deepopt/model-f9j1q6r5:v0',
                     '3':'omarcos/deepopt/model-29x4g8yu:v0'
                    },
                'order_0': ['/home/obanmarcos/Balseiro/DeepOPT/datasets/x14/140117_3dpf_lower tail_14', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x14/140315_3dpf_head_14', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x14/140117_3dpf_head_14', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x14/140114_5dpf_head_14', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x14/140519_5dpf_head_14', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x14/140315_1dpf_head_14', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x14/140114_5dpf_lower tail_14', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x14/140714_5dpf_head_14', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x14/140117_3dpf_body_14', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x14/140114_5dpf_upper tail_14', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x14/140117_3dpf_upper tail_14', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x14/140114_5dpf_body_14']
                },
                '18':
                {'models':
                {'0': 'omarcos/deepopt/model-24gv33q2:v0',
                     '1': 'omarcos/deepopt/model-1qs0uo7v:v0',
                     '2':'omarcos/deepopt/model-2cpkr2dl:v0',
                     '3':'omarcos/deepopt/model-2cpkr2dl:v0'
                    },
                'order_0':['/home/obanmarcos/Balseiro/DeepOPT/datasets/x18/140315_3dpf_head_18', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x18/140117_3dpf_lower tail_18', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x18/140117_3dpf_upper tail_18', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x18/140117_3dpf_body_18', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x18/140114_5dpf_upper tail_18', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x18/140114_5dpf_body_18', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x18/140114_5dpf_head_18', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x18/140114_5dpf_lower tail_18', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x18/140519_5dpf_head_18', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x18/140714_5dpf_head_18', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x18/140315_1dpf_head_18', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x18/140117_3dpf_head_18']
                },
                '22':
                {'models':
                {'0': 'omarcos/deepopt/model-3dp1wex6:v0',
                     '1': 'omarcos/deepopt/model-2jwf0rwa:v0',
                     '2':'omarcos/deepopt/model-1qtf5f8u:v0',
                     '3':'omarcos/deepopt/model-2nxos558:v0'
                },
                'order_0':['/home/obanmarcos/Balseiro/DeepOPT/datasets/x22/140315_3dpf_head_22', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x22/140114_5dpf_head_22', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x22/140519_5dpf_head_22', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x22/140117_3dpf_body_22', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x22/140114_5dpf_upper tail_22', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x22/140315_1dpf_head_22', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x22/140114_5dpf_lower tail_22', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x22/140714_5dpf_head_22', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x22/140117_3dpf_head_22', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x22/140117_3dpf_lower tail_22', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x22/140117_3dpf_upper tail_22', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x22/140114_5dpf_body_22']
                },
                '26':
                {'models':
                {'0': 'omarcos/deepopt/model-32wj43mf:v0',
                     '1': 'omarcos/deepopt/model-3kmtjdm4:v0',
                     '2':'omarcos/deepopt/model-3l028zex:v0',
                     '3':'omarcos/deepopt/model-2jnmr8t0:v0'
                    },
                'order_0': ['/home/obanmarcos/Balseiro/DeepOPT/datasets/x26/140114_5dpf_head_26', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x26/140315_3dpf_head_26', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x26/140117_3dpf_head_26', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x26/140519_5dpf_head_26', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x26/140117_3dpf_lower tail_26', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x26/140114_5dpf_upper tail_26', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x26/140714_5dpf_head_26', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x26/140117_3dpf_body_26', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x26/140117_3dpf_upper tail_26', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x26/140114_5dpf_lower tail_26', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x26/140315_1dpf_head_26', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x26/140114_5dpf_body_26']
                    },
                },
            'ssim':
                {'2':
                {'models': 
                    {'0':'omarcos/deepopt/model-1knoqwz4:v0',
                     '1':'omarcos/deepopt/model-2r2yp6pi:v0',
                     '2':'omarcos/deepopt/model-2r6xowyu:v0',
                     '3':'omarcos/deepopt/model-10ocv8c8:v0'
                    },
                'order_0': ['/home/obanmarcos/Balseiro/DeepOPT/datasets/x2/140519_5dpf_head_2', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x2/140117_3dpf_lower tail_2', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x2/140714_5dpf_head_2', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x2/140315_3dpf_head_2', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x2/140117_3dpf_head_2', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x2/140114_5dpf_head_2', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x2/140114_5dpf_upper tail_2', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x2/140117_3dpf_body_2', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x2/140315_1dpf_head_2', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x2/140117_3dpf_upper tail_2', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x2/140114_5dpf_body_2', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x2/140114_5dpf_lower tail_2']
                },
                '6': 
                {'models': 
                {'0':'omarcos/deepopt/model-pt8a7a9u:v0',
                     '1':'omarcos/deepopt/model-2m5ccwz7:v0',
                     '2':'omarcos/deepopt/model-2jk5121n:v0',
                     '3':'omarcos/deepopt/model-334pu3db:v0'
                    },
                'order_0':['/home/obanmarcos/Balseiro/DeepOPT/datasets/x6/140117_3dpf_lower tail_6', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x6/140714_5dpf_head_6', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x6/140114_5dpf_lower tail_6', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x6/140114_5dpf_upper tail_6', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x6/140117_3dpf_body_6', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x6/140519_5dpf_head_6', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x6/140117_3dpf_head_6', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x6/140315_3dpf_head_6', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x6/140114_5dpf_body_6','/home/obanmarcos/Balseiro/DeepOPT/datasets/x6/140315_1dpf_head_6', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x6/140117_3dpf_upper tail_6', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x6/140114_5dpf_head_6']
                },
                '10':
                {'models':
                {'0':'omarcos/deepopt/model-2aoiqvuu:v0',
                     '1':'omarcos/deepopt/model-18qqq8ov:v0',
                     '2':'omarcos/deepopt/model-32yat3q5:v0',
                     '3':'omarcos/deepopt/model-1r53j8ys:v0'
                     },
                'order_0':['/home/obanmarcos/Balseiro/DeepOPT/datasets/x10/140114_5dpf_upper tail_10', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x10/140315_3dpf_head_10', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x10/140117_3dpf_body_10', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x10/140315_1dpf_head_10', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x10/140117_3dpf_upper tail_10', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x10/140519_5dpf_head_10', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x10/140117_3dpf_head_10', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x10/140117_3dpf_lower tail_10', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x10/140114_5dpf_head_10', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x10/140714_5dpf_head_10', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x10/140114_5dpf_lower tail_10', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x10/140114_5dpf_body_10']    
                },
                '14':
                {'models':
                {'0':'omarcos/deepopt/model-2sftdt8l:v0',
                     '1':'omarcos/deepopt/model-bp5tv3lf:v0',
                     '2':'omarcos/deepopt/model-33khl4s3:v0',
                     '3':'omarcos/deepopt/model-26eu06np:v0'
                    },
                'order_0':['/home/obanmarcos/Balseiro/DeepOPT/datasets/x14/140117_3dpf_lower tail_14', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x14/140315_3dpf_head_14', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x14/140117_3dpf_head_14', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x14/140114_5dpf_head_14', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x14/140519_5dpf_head_14', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x14/140315_1dpf_head_14', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x14/140114_5dpf_lower tail_14', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x14/140714_5dpf_head_14', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x14/140117_3dpf_body_14', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x14/140114_5dpf_upper tail_14', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x14/140117_3dpf_upper tail_14', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x14/140114_5dpf_body_14']
                },
                '18':
                {'models':
                {'0': 'omarcos/deepopt/model-39q68n8a:v0',
                     '1': 'omarcos/deepopt/model-1lorqlgs:v0',
                     '2':'omarcos/deepopt/model-3rkm7lhb:v0',
                     '3':'omarcos/deepopt/model-2e890ftb:v0'
                    },
                'order_0':['/home/obanmarcos/Balseiro/DeepOPT/datasets/x18/140315_3dpf_head_18', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x18/140117_3dpf_lower tail_18', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x18/140117_3dpf_upper tail_18', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x18/140117_3dpf_body_18', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x18/140114_5dpf_upper tail_18', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x18/140114_5dpf_body_18', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x18/140114_5dpf_head_18', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x18/140114_5dpf_lower tail_18', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x18/140519_5dpf_head_18', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x18/140714_5dpf_head_18', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x18/140315_1dpf_head_18', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x18/140117_3dpf_head_18']
                },
                '22':
                {'models':
                {'0': 'omarcos/deepopt/model-2srs5uf0:v0',
                     '1': 'omarcos/deepopt/model-sy28fr8u:v0',
                     '2':'omarcos/deepopt/model-3kx90j5f:v0',
                     '3':'omarcos/deepopt/model-nbhkzvx1:v0'
                    },
                'order_0':['/home/obanmarcos/Balseiro/DeepOPT/datasets/x22/140315_3dpf_head_22', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x22/140114_5dpf_head_22', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x22/140519_5dpf_head_22', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x22/140117_3dpf_body_22', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x22/140114_5dpf_upper tail_22', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x22/140315_1dpf_head_22', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x22/140114_5dpf_lower tail_22', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x22/140714_5dpf_head_22', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x22/140117_3dpf_head_22', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x22/140117_3dpf_lower tail_22', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x22/140117_3dpf_upper tail_22', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x22/140114_5dpf_body_22']
                },
                '26':
                {'models':
                {'0': 'omarcos/deepopt/model-2srs5uf0:v0',
                     '1': 'omarcos/deepopt/model-sy28fr8u:v0',
                     '2':'omarcos/deepopt/model-3kx90j5f:v0',
                     '3':'omarcos/deepopt/model-nbhkzvx1:v0'
                },
                'order_0':['/home/obanmarcos/Balseiro/DeepOPT/datasets/x26/140114_5dpf_head_26', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x26/140315_3dpf_head_26', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x26/140117_3dpf_head_26', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x26/140519_5dpf_head_26', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x26/140117_3dpf_lower tail_26', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x26/140114_5dpf_upper tail_26', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x26/140714_5dpf_head_26', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x26/140117_3dpf_body_26', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x26/140117_3dpf_upper tail_26', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x26/140114_5dpf_lower tail_26', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x26/140315_1dpf_head_26', '/home/obanmarcos/Balseiro/DeepOPT/datasets/x26/140114_5dpf_body_26']
                }
                }
            }

def runs(testing_options):
# Model dictionary
    if use_default_model_dict == True:
        #U-Net model
        resnet_options_dict = {'number_layers': 5,
                        'kernel_size':3,
                        'features':64,
                        'in_channels':1,
                        'out_channels':1,
                        'stride':1, 
                        'use_batch_norm': True,
                        'init_method': 'xavier'}

    # Model parameters
        modl_dict = {'use_torch_radon': False,
                    'metric': 'psnr',
                    'K_iterations' : 6,
                    'number_projections_total' : 720,
                    'acceleration_factor': 10,
                    'image_size': 100,
                    'lambda': 0.05,
                    'use_shared_weights': True,
                    'denoiser_method': 'resnet',
                    'resnet_options': resnet_options_dict,
                    'in_channels': 1,
                    'out_channels': 1}

        # Training parameters
        loss_dict = {'loss_name': 'psnr',
                    'psnr_loss': torch.nn.MSELoss(reduction = 'mean'),
                    'ssim_loss': SSIM(data_range=1, size_average=True, channel=1),
                    'msssim_loss': MSSSIM(kernel_size = 1), 
                    'l1_loss': torch.nn.L1Loss(reduction = 'mean')}

        # Optimizer parameters
        optimizer_dict = {'optimizer_name': 'Adam+Tanh',
                        'lr': 1e-4}

        # System parameters
        # System parameters
        model_system_dict = {'acc_factor_data': 1,
                        'use_normalize': True,
                        'optimizer_dict': optimizer_dict,
                        'kw_dictionary_modl': modl_dict,
                        'loss_dict': loss_dict, 
                        'method':'modl',                       
                        'track_train': True,
                        'track_val': True,
                        'track_test': True,
                        'max_epochs': 20, 
                        'track_alternating_admm':False,
                        'tv_iters': 40,
                        'title': 'HyperParams_Search',
                        'metrics_folder': where_am_i('metrics'),
                        'models_folder': where_am_i('models'),
                        'track_alternating_admm': False,         
                        'track_alternating_twist': False,
                        'track_unet': False}

    # PL Trainer and W&B logger dictionaries
    if use_default_trainer_dict == True:
                
        logger_dict = {'project':'deepopt',
                        'entity': 'omarcos', 
                        'log_model': 'model'}
 
        lightning_trainer_dict = {'max_epochs': 20,
                                  'log_every_n_steps': 10,
                                  'check_val_every_n_epoch': 1,
                                  'gradient_clip_val': 0.3, 
                                  'gradient_clip_algorithm':"value",
                                  'accelerator' : 'gpu', 
                                  'devices' : 1,
                                  'fast_dev_run' : False,
                                  'default_root_dir': where_am_i('models')}

        trainer_dict = {'lightning_trainer_dict': lightning_trainer_dict,
                        'use_k_folding': True, 
                        'track_checkpoints': True,
                        'epoch_number_checkpoint': 10,
                        'use_swa' : False,
                        'use_accumulate_batches': False,
                        'k_fold_number_datasets': 3,
                        'use_logger' : True,
                        'resume':'allow',
                        'logger_dict': logger_dict,
                        'track_default_checkpoints'  : True,
                        'use_auto_lr_find': False,
                        'batch_accumulate_number': 3,
                        'use_mixed_precision': False,
                        'batch_accumulation_start_epoch': 0, 
                        'profiler': profiler,
                        'restore_fold': False,
                        'fold_number_restore': 2,
                        'acc_factor_restore': 22}

    # Dataloader dictionary
    if use_default_dataloader_dict == True:
        
        # data_transform = T.Compose([T.ToTensor()])
        data_transform = None                                    
        
        dataloader_dict = {'datasets_folder': where_am_i('datasets'),
                           'number_volumes' : 0,
                           'experiment_name': 'Bassi',
                           'img_resize': 100,
                           'load_shifts': True,
                           'save_shifts':False,
                           'number_projections_total': 720,
                           'acceleration_factor':20,
                           'train_factor' : 0.8, 
                           'val_factor' : 0.2,
                           'test_factor' : 0.2,     
                           'batch_size' : 10, 
                           'sampling_method' : 'equispaced-linear',
                           'shuffle_data' : True,
                           'data_transform' : data_transform,
                           'num_workers' : 0}
    
    if 'load_unet' in testing_options:

        acceleration, fold = '26', '0'
    
        model_system_dict['load_model'] = True
        model_system_dict['load_path'] = 'Unet_FA{}_Kfold{}'.format(acceleration, fold)
        dataloader_dict['acceleration_factor'] = int(acceleration)

        model = UNetReconstructor(model_system_dict) 

        trainer_system = trutils.TrainerSystem(trainer_dict, dataloader_dict, model_system_dict)

        datasets_list = model_dict['psnr'][acceleration]['order_0']
        trainer_system.set_datasets_list(datasets_list)   

        _,_, test_dataloader = trainer_system.generate_K_folding_dataloader()

        trainer = trainer_system.create_trainer()
                # Train
        test_dict = trainer.test(model=model, 
                        dataloaders = test_dataloader)

        print(test_dict)

    if 'train_psnr' in testing_options:
        
        model_system_dict['loss_dict']['loss_name'] = 'psnr'
        accelerations = [str(i) for i in range(26,-2,-4)]

        for acceleration in accelerations:
            
            dataloader_dict['acceleration_factor'] = int(acceleration)
            
            # Rotation of dataset
            trainer_system = trutils.TrainerSystem(trainer_dict, dataloader_dict, model_system_dict)

            datasets_list = model_dict['psnr'][acceleration]['order_0']
            trainer_system.set_datasets_list(datasets_list)    

            for fold in range(len(datasets_list)//trainer_dict['k_fold_number_datasets']):

                model_system_dict['load_path'] = ''
                model_system_dict['save_path'] = 'MoDL_FA{}_Kfold{}'.format(acceleration, fold)

                wandb.init(project = 'MoDL', name = model_system_dict['save_path'])

                # Model U-Net
                model = MoDLReconstructor(model_system_dict)    

                wandb.watch(model)

                train_dataloader, val_dataloader, test_dataloader = trainer_system.generate_K_folding_dataloader()
                trainer_system.current_fold += 1
                
                trainer = trainer_system.create_trainer()
                # Train
                trainer.fit(model=model, 
                        train_dataloaders= train_dataloader,
                        val_dataloaders = val_dataloader)

                model.save_model()

                test_dict = trainer.test(model=model, 
                        dataloaders = test_dataloader)

                torch.cuda.empty_cache()
                del train_dataloader, val_dataloader, model.model
                del model
                wandb.finish()


if __name__ == '__main__':

    train_options = []

    parser = argparse.ArgumentParser(description='Do K-folding with different networks')

    parser.add_argument('--train_psnr', help = 'Train w/L1 loss with optimal hyperparameters', action="store_true")
    parser.add_argument('--train_ssim', help = 'Train w/SSIM loss with optimal hyperparameters', action="store_true")
    parser.add_argument('--load_unet', help = 'Load U-Net', action="store_true")
    
    args = parser.parse_args()

    if args.train_psnr:

        print('Training MODL with PSNR loss...')
        train_options.append('train_psnr')
    
    runs(train_options)
