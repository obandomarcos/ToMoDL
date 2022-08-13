'''
Testing functionalities of dataloading_utilities

author: obanmarcos
'''
import os
import os, sys

sys.path.append('/home/obanmarcos/Balseiro/DeepOPT/')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utilities import dataloading_utilities as dlutils
from utilities.folders import *
from utilities import model_utilities as modutils
import torch
import cv2

def test_dataloader():

    # Using CPU or GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    folder_paths = [f140115_1dpf]

    zebra_datasets = dlutils.ZebraDataloader(folder_paths)

    # 1 - Load datasets
    # 1a - Check ZebraDataset.load_images()
    zebra_dataset_test = zebra_datasets[folder_paths[0]]
    zebra_dataset_test.load_images()
    
    # Print main attributes after load_images
    print(zebra_dataset_test.registered_dataset)
    print(zebra_dataset_test.registered_volume['head'].max())
    print(zebra_dataset_test.image_volume['head'].max())

    # Print registered
    cv2.imwrite('/home/obanmarcos/Balseiro/DeepOPT/Tests/Tests_Dataloader_Utilities/'+'Test_Image.jpg', 255.0*zebra_dataset_test._normalize_image(zebra_dataset_test.image_volume['head'][0,...]))

    # Print non-registered
    cv2.imwrite('/home/obanmarcos/Balseiro/DeepOPT/Tests/Tests_Dataloader_Utilities/'+'Test_Image_Registered.jpg', 255.0*zebra_dataset_test._normalize_image(zebra_dataset_test.registered_volume['head'][0,...]))

if __name__ == '__main__':

    test_dataloader()