import os
import os, sys
from config import * 

sys.path.append(where_am_i())

from utilities import dataloading_utilities as dlutils
from torch.utils.data import DataLoader
from utilities.folders import *
import pandas as pd
import numpy as np
import pickle
import argparse
import matplotlib.pyplot as plt
import skimage.measure    
import seaborn as sns

pickle_file = './logs/14-IntensityStructure.pkl'

def variance_datasets():

    acceleration_factor = 22
    dataset_list_names = ['140117_3dpf_lower tail_22', '140114_5dpf_head_22', '140519_5dpf_head_22', '140117_3dpf_body_22', '140114_5dpf_upper tail_22', '140315_1dpf_head_22', '140114_5dpf_lower tail_22', '140714_5dpf_head_22', '140117_3dpf_head_22', '140117_3dpf_lower tail_22', '140117_3dpf_upper tail_22', '140114_5dpf_body_22'] # paths

    dataset_list_paths = [datasets_folder+'x{}/'.format(acceleration_factor)+x for x in dataset_list_names] 

    if os.path.exists(pickle_file):
        
        print('Loading pickle...')
        
        with open(pickle_file, 'rb') as f:

            variance = pickle.load(f)

    else:

        variance = {}

    # Run testing over slices/everything
    for dataset_name, dataset_path in zip(dataset_list_names, dataset_list_paths):

        fish_part = dataset_name.split('_')[-2]
        fish_dpf = dataset_name.split('_')[-3]

        dataset_dict = {'root_folder' : dataset_path, 
                        'acceleration_factor' : acceleration_factor,
                        'transform' : None}

        test_dataset = dlutils.ReconstructionDataset(**dataset_dict)    

        test_dataloader = DataLoader(test_dataset, 
                                    batch_size = 1,
                                    shuffle = False,
                                    num_workers = 8)
        variance[dataset_name] = []

        for (us_uf, us_fil, fs_fil) in test_dataloader:
            
            us_uf_var = entropy = skimage.measure.shannon_entropy(us_uf.cpu().numpy())
            us_fil_var = skimage.measure.shannon_entropy(us_fil.cpu().numpy())
            fs_fil_var = skimage.measure.shannon_entropy(fs_fil.cpu().numpy())

            image_variances = [us_uf_var, us_fil_var, fs_fil_var]
            print('Intensity variances for: Us-Unfilt {}, Us-Filt {}, Fs-Filt {}'.format(*image_variances))
            variance[dataset_name].append(image_variances)
        
        variance[dataset_name] = np.array(variance[dataset_name])

    with open(pickle_file, 'wb') as f:

        pickle.dump(variance, f)

def plot_intensities():

    dataset_names = ['140117_3dpf_lower tail_22', '140114_5dpf_head_22', '140519_5dpf_head_22', '140117_3dpf_body_22', '140114_5dpf_upper tail_22', '140315_1dpf_head_22', '140114_5dpf_lower tail_22', '140714_5dpf_head_22', '140117_3dpf_head_22', '140117_3dpf_lower tail_22', '140117_3dpf_upper tail_22', '140114_5dpf_body_22'] 

    with open(pickle_file, 'rb') as f:

        variance = pickle.load(f)
    
    fig, axs = plt.subplots(3,len(dataset_names)//3, figsize = (20, 16))

    axs = axs.flatten()

    for dataset_name, ax in zip(dataset_names, axs):

        ax.plot(variance[dataset_name][:,0], label = 'Undersampled unfiltered')
        ax.plot(variance[dataset_name][:,1], label = 'Undersampled filtered')
        ax.plot(variance[dataset_name][:,2], label = 'Fully sampled filtered')
    
        ax.set_xlabel('# of Slice')
        ax.set_title(dataset_name)
        ax.legend()
        sns.despine()

    fig.savefig('logs/14-IntensityStructure_AllDatasets.pdf'.format(dataset_name), bbox_inches = 'tight')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--var', help='Calculate Variance', action='store_true')
    parser.add_argument('--plot', help='Plot Variance', action='store_true')

    args = vars(parser.parse_args())
    
    if args['var'] == True:
        print('Calculating variances...')
        variance_datasets()

    if args['plot'] == True:
        
        print('Plotting variances...')
        plot_intensities()  
