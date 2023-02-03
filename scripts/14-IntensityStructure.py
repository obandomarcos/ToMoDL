import os
import os, sys
from config import * 

sys.path.append(where_am_i())

from pathlib import Path
from utilities import dataloading_utilities as dlutils
from scipy.signal import correlate
from torch.utils.data import DataLoader
from utilities.folders import *
import pandas as pd
import numpy as np
import pickle
import argparse
import matplotlib.pyplot as plt
import skimage.measure  
from matplotlib.gridspec import GridSpec  
import scipy
import seaborn as sns

import matplotlib
matplotlib.use('Agg')

pickle_file = './logs/14-IntensityStructure.pkl'
dataframe_pickle = './logs/test_dataframe_x22.pkl'

def variance_datasets():

    acceleration_factor = 22
    dataset_list_names = ['140315_3dpf_head_22', '140114_5dpf_head_22', '140519_5dpf_head_22', '140117_3dpf_body_22', '140114_5dpf_upper tail_22', '140315_1dpf_head_22', '140114_5dpf_lower tail_22', '140714_5dpf_head_22', '140117_3dpf_head_22', '140117_3dpf_lower tail_22', '140117_3dpf_upper tail_22', '140114_5dpf_body_22'] # paths

    dataset_list_paths = [where_am_i('datasets')+'x{}/'.format(acceleration_factor)+x for x in dataset_list_names] 

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
            
            us_uf_var = np.std(us_uf.cpu().numpy())
            us_fil_var = np.std(us_fil.cpu().numpy())
            fs_fil_var = np.std(fs_fil.cpu().numpy())

            image_variances = [us_uf_var, us_fil_var, fs_fil_var]
            print('Intensity variances for: Us-Unfilt {}, Us-Filt {}, Fs-Filt {}'.format(*image_variances))
            variance[dataset_name].append(image_variances)
        
        variance[dataset_name] = np.array(variance[dataset_name])

    with open(pickle_file, 'wb') as f:

        pickle.dump(variance, f)

def plot_intensities():

    dataset_names = ['140114_5dpf_head_22', '140519_5dpf_head_22', '140117_3dpf_head_22', '140315_3dpf_head_22', '140315_1dpf_head_22', '140714_5dpf_head_22','140117_3dpf_body_22',  '140114_5dpf_body_22', '140114_5dpf_upper tail_22', '140117_3dpf_upper tail_22', '140114_5dpf_lower tail_22','140117_3dpf_lower tail_22']
     

    with open(pickle_file, 'rb') as f:

        variance = pickle.load(f)

    dataset_names_alt = [ '140117_3dpf_lower tail_22', '140117_3dpf_upper tail_22', '140117_3dpf_body_22','140117_3dpf_head_22']
    parts = ['Cola Inferior', 'Cola Superior', 'Cuerpo', 'Cabeza']
    fig = plt.figure(figsize = (12, 8))
    gs = GridSpec(2,3, figure = fig)

    variance_ax = fig.add_subplot(gs[0,:])
    histogram_ax = fig.add_subplot(gs[1,2])
    gradient_ax =  fig.add_subplot(gs[1,0:2])

    for enum, (dataset_name, part) in enumerate(zip(dataset_names_alt, parts)): 

        variance_ax.plot(variance[dataset_name][:,0], label = part)
        # ax.plot(variance[dataset_name][:,1], label = 'FBP - submuestreada')
        # ax.plot(variance[dataset_name][:,2], label = 'FBP - muestreo completo')
        variance_ax.legend(loc = 2)

        gradient_ax.plot(np.diff(variance[dataset_name][:,0]), label = part)
        histogram_ax.hist(np.diff(variance[dataset_name][:,0]), label = part, alpha = 0.4, bins = 60, orientation='horizontal', density= True)

        variance_ax.set_xlabel('Corte i-ésimo')
        gradient_ax.set_xlabel('Corte i-ésimo')

        variance_ax.set_ylabel(r'$\sigma^i_{\mathrm{I}}$')
        variance_ax.text(-120 , 0.6, 'a)')
        gradient_ax.set_ylabel(r'$\Delta \sigma^i_{\mathrm{I}}$')

        histogram_ax.set_xlabel(r'$P(\Delta \sigma_I)$')
        histogram_ax.set_ylim(-0.05, 0.05)
        gradient_ax.set_ylim(-0.05, 0.05)
        gradient_ax.text(-160 , 0.05, 'b)')

        histogram_ax.set_yticks([])
        histogram_ax.set_yticklabels([])
        histogram_ax.text(-15 , 0.05, 'c)')
        
        handles, labels = gradient_ax.get_legend_handles_labels()
        variance_ax.legend(handles[::-1], labels[::-1])
        sns.despine()

    fig.savefig('logs/14-IntensityStructure_AllDatasets_HistPlot.pdf'.format(dataset_name), bbox_inches = 'tight')

    

def datacode_string(dataset_name):

    vol_code= {'140519':'A', '140117':'B', '140714':'C', '140315':'D', '140114':'E'}
    dpf_code = {'5dpf':'5 días', '3dpf':'3 días', '1dpf':'1 día'}
    sec_code= {'head':'Cabeza', 'body':'Cuerpo', 'upper tail':'Cola Superior', 'lower tail': 'Cola Inferior'}

    vol = dataset_name.split('_')[-4]
    dpf = dataset_name.split('_')[-3]
    sec = dataset_name.split('_')[-2]

    return '{}'.format(sec_code[sec])
    # return 'Volúmen {} \n- {} - {}'.format(vol_code[vol], dpf_code[dpf], sec_code[sec])

def correlate_intensity():

    acceleration_factor = 22
    dataset_names = ['140117_3dpf_lower tail_22', '140117_3dpf_upper tail_22', '140117_3dpf_body_22', '140117_3dpf_head_22']

    with open(pickle_file, 'rb') as f:

        variance = pickle.load(f)
    
    dataframe = pd.read_pickle(dataframe_pickle)

    metrics = ['psnr', 'ssim']
    model_metrics = ['test/{}', 'test/{}_fbp',]
    model_metric_names = ['ToMoDL', 'FBP']

    for metric in metrics:

        i = 0

        fig, axs = plt.subplots(1,4, figsize = (16,6), sharex = True, sharey = True)


        axs = axs.flatten()

        for model_metric, model_metric_name in zip(model_metrics, model_metric_names):
        
            for enum, (dataset_name, ax) in enumerate(zip(dataset_names, axs)):
                
                
                datacode = dataset_name.split('_')[-4].split('/')[-1]
                fish_part = dataset_name.split('_')[-2]
                fish_dpf = dataset_name.split('_')[-3]
                
                entropy = dataframe[(dataframe['datacode'] == datacode) & (dataframe['fish_part'] == fish_part) & (dataframe['fish_dpf'] == fish_dpf)][model_metric.format(metric)].apply(pd.Series).astype(np.float64).stack().reset_index(drop=True)  

                cross_corr = np.log10(correlate(entropy, variance[dataset_name][:,0], mode = 'same'))
                slices = np.arange(len(cross_corr))-len(cross_corr)//2
                pearson = round(scipy.stats.pearsonr(entropy, variance[dataset_name][:,0]).statistic, 3)

                # Not entropy!!
                ax.scatter(entropy[:-1], np.diff(np.sqrt(variance[dataset_name][:,0])), label = model_metric_name.format(metric.upper())+'\nr = {}'.format(pearson), alpha = 0.6)

                if model_metric == 'test/{}':

                    print(np.argmin(entropy))

                ax.set_xlabel('PSNR [dB]', fontsize = 18)
                ax.set_ylim(-0.08, 0.08)
                if enum % 4 == 0:
                    ax.set_ylabel(r'$\Delta\sigma^i_{\mathrm{I}}$', fontsize = 18)

                ax.legend(loc = 2, fontsize = 18)
                    
                # ax.set_title(datacode_string(dataset_name))
                sns.despine()     

        fig.savefig('logs/14-EntropyMetrics_{}_AllDatasets.pdf'.format(metric), bbox_inches = 'tight')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--var', help='Calculate Variance', action='store_true')
    parser.add_argument('--plot', help='Plot Variance', action='store_true')
    parser.add_argument('--corr', help='Correlate entropy and performance', action='store_true')
    parser.add_argument('--check_psnr', help='Check PSNR', action='store_true')

    args = vars(parser.parse_args())
    
    if args['check_psnr'] == True:
        
        print('Checking PSNR...')
        check_psnr_patches()
        
    if args['var'] == True:
        print('Calculating variances...')
        variance_datasets()

    if args['plot'] == True:
        
        print('Plotting variances...')
        plot_intensities()  
    
    if args['corr'] == True:
        
        print('Correlating intensities...')
        correlate_intensity()