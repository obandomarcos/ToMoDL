'''
Evaluate K-Fold models from artifacts
author: obanmarcos
'''
import os, sys
from config import *

sys.path.append(where_am_i())

import pytorch_lightning as pl
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from training import train_utilities as trutils
from utilities import dataloading_utilities as dlutils
from utilities.folders import *



from models.models_system import MoDLReconstructor
import torch

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from torchvision import transforms as T
from pytorch_msssim import SSIM

import wandb

# Options for folding menu
use_default_model_dict = True
use_default_dataloader_dict = True
use_default_trainer_dict = True

def eval_models(testing_options):

    if 'load_run' in testing_options:
        
        run = wandb.init()
        artifact = run.use_artifact('omarcos/deepopt/model-1ud4xx3w:v0', type='model')
        artifact_dir = artifact.download()

        print('Artifact directory:\n')
        print(artifact_dir)

        print('Artifact:\n')
        print(artifact)

    return


if __name__ == '__main__':

    testing_options = []

    parser = argparse.ArgumentParser(description='Load run')

    parser.add_argument('--load_run', help = 'Train w/PSNR loss with optimal hyperparameters', action="store_true")
    
    args = parser.parse_args()

    if args.load_run:
        
        print('Loading runs...')
        testing_options.append('load_run')
    
    eval_models(testing_options)