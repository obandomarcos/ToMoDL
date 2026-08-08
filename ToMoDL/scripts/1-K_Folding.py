"""
K-Folding script
author: obanmarcos
"""

import os
import os, sys
from config import *

sys.path.append(where_am_i())

import lightning as pl
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utilities import dataloading_utilities as dlutils
from utilities.folders import *

from training import train_utilities as trutils

from models.models_system import MoDLReconstructor
import torch


from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from torchvision import transforms as T
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics import MeanSquaredError as MSE
from torch.nn import L1Loss

# from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure as MSSSIM

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["PYTORCH_USE_CUDA_DSA"] = "1"
torch.set_float32_matmul_precision("high")
# Options for folding menu
use_default_model_dict = True
use_default_dataloader_dict = True
use_default_trainer_dict = True
device_id = 0
device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
max_epochs = 50
image_size_train = 128
print(where_am_i("datasets"))
acc_factor = 20


def runs(testing_options):

    # Model dictionary
    if use_default_model_dict == True:
        # ResNet dictionary parameters
        resnet_options_dict = {
            "number_layers": 8,
            "kernel_size": 3,
            "features": 64,
            "in_channels": 1,
            "out_channels": 1,
            "stride": 1,
            "use_batch_norm": True,
            "init_method": "xavier",
        }

        # Model parameters
        modl_dict = {
            "use_torch_radon": True,
            "metric": "psnr",
            "K_iterations": 6,
            "number_projections": 720,
            "lambda": 0.15,
            "use_shared_weights": True,
            "denoiser_method": "U-Net",
            "resnet_options": resnet_options_dict,
            "in_channels": 1,
            "out_channels": 1,
            "iter_conjugate": 5,
        }

        # Training parameters
        loss_dict = {
            "loss_name": "mse",
            "psnr_loss": PSNR().to(device),
            "mse_loss": MSE().to(device),
            "l1_loss": L1Loss().to(device),
            "ssim_loss": SSIM().to(device),
            "msssim_loss": MSSSIM(kernel_size=1).to(device),
        }

        # Optimizer parameters
        # optimizer_dict = {"optimizer_name": "Adam+Tanh", "lr": 1e-4}
        optimizer_dict = {"optimizer_name": "NAdam", "lr": 4e-4}

        # System parameters
        model_system_dict = {
            "acc_factor_data": 1,
            "use_normalize": True,
            "optimizer_dict": optimizer_dict,
            "kw_dictionary_modl": modl_dict,
            "loss_dict": loss_dict,
            "method": "modl",
            "track_train": True,
            "track_val": True,
            "track_test": True,
            "max_epochs": max_epochs,
            "save_model": True,
            "load_path": "",
            "save_path": "MoDL_K_fold_{}",
            "track_alternating_admm": False,
            "title": "HyperParams_Search",
            "metrics_folder": where_am_i("metrics"),
            "models_folder": where_am_i("models"),
            "track_alternating_admm": False,
            "track_alternating_twist": False,
            "track_unet": False,
        }

    # PL Trainer and W&B logger dictionaries
    if use_default_trainer_dict == True:

        logger_dict = {
            "project": "ToMoDL",
            # 'entity': 'omarcos',
            "log_model": True,
        }

        lightning_trainer_dict = {
            "max_epochs": max_epochs,
            "log_every_n_steps": 10,
            "check_val_every_n_epoch": 1,
            "gradient_clip_val": 1,
            "accelerator": "gpu",
            "devices": [device_id],
            "fast_dev_run": False,
            "default_root_dir": where_am_i("models"),
        }

        profiler = None
        # profiler = SimpleProfiler(dirpath = './logs/', filename = 'Test_training_profile_pytorch')
        # profiler = PyTorchProfiler(dirpath = './logs/', filename = 'Test_training_profile_pytorch')
        trainer_dict = {
            "lightning_trainer_dict": lightning_trainer_dict,
            "use_k_folding": True,
            "track_checkpoints": True,
            "epoch_number_checkpoint": 5,
            "use_swa": False,
            "use_accumulate_batches": False,
            "k_fold_number_datasets": 2,
            "use_logger": True,
            "logger_dict": logger_dict,
            "track_default_checkpoints": True,
            "use_auto_lr_find": False,
            "batch_accumulate_number": 3,
            "use_mixed_precision": False,
            "precision": "16-mixed",
            "batch_accumulation_start_epoch": 0,
            "profiler": profiler,
            "restore_fold": False,
            "resume": False,
        }

    # Dataloader dictionary
    if use_default_dataloader_dict == True:

        # data_transform = T.Compose([T.ToTensor()])
        data_transform = None

        dataloader_dict = {
            "datasets_folder": f"{where_am_i('datasets')}full_fish_{image_size_train}/",
            "number_volumes": 0,
            "experiment_name": "Bassi",
            "image_size": image_size_train,
            "load_shifts": True,
            "save_shifts": False,
            "number_projections_total": 720,
            "acceleration_factor": 20,
            "train_factor": 0.8,
            "val_factor": 0.2,
            "test_factor": 0.2,
            "batch_size": 8,
            "sampling_method": "equispaced-linear",
            "shuffle_data": True,
            "data_transform": data_transform,
            "num_workers": 8,
            "use_subset_by_part": False,
        }

    acc_factor = 20
    dataloader_dict["acceleration_factor"] = acc_factor
    model_system_dict["kw_dictionary_modl"]["acceleration_factor"] = acc_factor

    # Create Custom trainer
    if "train_ssim" in testing_options:

        with torch.autograd.set_detect_anomaly(True):

            model_system_dict["loss_dict"]["loss_name"] = "ssim"

            trainer = trutils.TrainerSystem(trainer_dict, dataloader_dict, model_system_dict)
            trainer.k_folding()

    if "train_msssim" in testing_options:

        with torch.autograd.set_detect_anomaly(True):

            model_system_dict["loss_dict"]["loss_name"] = "msssim"

            trainer = trutils.TrainerSystem(trainer_dict, dataloader_dict, model_system_dict)
            trainer.k_folding()

    if "train_psnr" in testing_options:

        model_system_dict["loss_dict"]["loss_name"] = "psnr"

        trainer = trutils.TrainerSystem(trainer_dict, dataloader_dict, model_system_dict)
        trainer.k_folding()

    if "train_mse" in testing_options:

        model_system_dict["loss_dict"]["loss_name"] = "mse"

        trainer = trutils.TrainerSystem(trainer_dict, dataloader_dict, model_system_dict)
        trainer.k_folding()


if __name__ == "__main__":

    k_folding_options = []

    parser = argparse.ArgumentParser(description="Do K-folding with different networks")
    parser.add_argument("--train_mse", help="Train w/MSE loss with optimal hyperparameters", action="store_true")
    parser.add_argument("--train_psnr", help="Train w/PSNR loss with optimal hyperparameters", action="store_true")
    parser.add_argument("--train_ssim", help="Train w/SSIM loss with optimal hyperparameters", action="store_true")
    parser.add_argument("--train_msssim", help="Train w/MS-SSIM loss with optimal hyperparameters", action="store_true")

    args = parser.parse_args()
    args.train_mse = True

    if args.train_mse:

        print("Training MODL with MSE loss...")
        k_folding_options.append("train_mse")

    if args.train_psnr:

        print("Training MODL with PSNR loss...")
        k_folding_options.append("train_psnr")

    if args.train_ssim:

        print("Training MODL with SSIM loss...")
        k_folding_options.append("train_ssim")

    if args.train_msssim:

        print("Training MODL with MS-SSIM loss...")
        k_folding_options.append("train_msssim")

    runs(k_folding_options)
