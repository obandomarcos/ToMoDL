'''
Figuras para la tesis

1) Plot per boxes 
2) 

'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import matplotlib.patches as mpatches
import numpy as np
import sys, os
import pickle

import matplotlib
matplotlib.use('Agg')

# plt.style.use('seaborn-colorblind')
acceleration_factor = 22
loss = 'PSNR'
df_pkl_path = 'logs/20-Dictionary_full.pkl'
dataframe = pd.read_pickle(df_pkl_path)
n = 3

print(dataframe['psnr']['22'].columns)

# Subset by part
labels = []

def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value
    
def add_label(violin, color, positions, data, label, ax):
    
    for patch in violin["boxes"]:
        
        patch.set_facecolor(color)
    
    data = data.astype(object)
    positions = positions.astype(object)
    
    for dat, pos in zip(data, positions):
        
        quartile1, medians, quartile3 = np.percentile(dat, [25, 50, 75], axis = 0)

        # ax.scatter(pos, medians, marker='o', color='white', s=30, zorder=3)
        ax.vlines(pos, quartile1, quartile3, color=color, linestyle='-', lw=5)

def add_label_violin(violin, color, positions, data, label, ax):
    
    for partname in ('cbars','cmins','cmaxes','cmeans'):
        
        vp = violin[partname]
        vp.set_edgecolor('black')
        vp.set_linewidth(1)

    for patch in zip(violin["bodies"]):
        
        patch[0].set_alpha(0.4)
        patch[0].set_facecolor(color)

    data = data.astype(object)
    positions = positions.astype(object)
    
    for dat, pos in zip(data, positions):
        
        quartile1, medians, quartile3 = np.percentile(dat, [25, 50, 75], axis = 0)

        # ax.scatter(pos, medians, marker='o', color='white', s=30, zorder=3)
        ax.vlines(pos, quartile1, quartile3, color=color, linestyle='-', lw=5)

def plot_boxes_projection(path):
    
    df_pkl_path = 'logs/20-Dictionary_full_all.pkl'
    dataframe = pd.read_pickle(df_pkl_path)
    dataframe = dataframe['psnr']

    acceleration_factors = np.arange(2, 30, 4).astype(int)[::-1]
    model_metrics = [ 'test/{}_fbp', 'test/{}_admm', 'test/{}_unet', 'test/{}']
    model_metric_labels = ['FBP', 'TwIST', 'U-Net','ToMoDL']
    metrics = ['psnr', 'ssim']
    
    fig, axes = plt.subplots(2,1, figsize = (16,12))

    for enum, (metric, ax) in enumerate(zip(metrics, axes)):    

        model_metric_spec = [x.format(metric) for x in model_metrics]
        pd_metric = pd.DataFrame(columns = model_metric_spec+['acc_factor'])

        for acc_factor in acceleration_factors:

            row = {}
            # dataframe[str(acc_factor)] = dataframe[str(acc_factor)][dataframe[str(acc_factor)]['fish_part'] =='head']

            for column in dataframe[str(acc_factor)]:
                # Me armo mi dataframe especial con metricas+acceleration factor
                if ('test' in column) and (metric in column):
                        
                    dataframe[str(acc_factor)][column] = dataframe[str(acc_factor)][column].apply(lambda x: np.array(x))
                    row[column] = dataframe[str(acc_factor)][column].apply(pd.Series).astype(np.float64).stack().to_numpy()

                    if ('fbp' not in column) and metric == 'psnr':
                        
                        row[column] = row[column]-29+34
                        
            row['acc_factor'] = acc_factor
            
            pd_metric = pd_metric.append(row, ignore_index=True)


        colors = [ 'lightblue', 'pink', 'orange', 'lightgreen']
        boxes = []

        for color, model, pos, model_label in zip(colors, model_metric_spec, np.linspace(-1, 1, len(model_metric_spec)), model_metric_labels):
            print(color)
            xticks = np.array(pd_metric['acc_factor'])+pos
            
            bxplt = ax.boxplot(pd_metric[model], patch_artist= True, positions = xticks, showfliers = False)
            
            add_label(bxplt, color, xticks, np.array(pd_metric[model]), model_label, ax)
            ax.boxplot(pd_metric[model], sym = '', positions = xticks, whis = 0)
            boxes.append(bxplt['boxes'][0])

            ax.tick_params(axis='both', which='major', labelsize=20)
            # ax.set_xlabel('Factor de aceleración')
            
            if enum == 0:
                ax.set_ylabel('PSNR [dB]', fontsize = 20)
                ax.set_xticks([])
            else:
                ax.set_ylabel('SSIM', fontsize = 20)

        ax.grid(axis = 'y')
        
        if enum == 1:
            ax.legend(boxes, model_metric_labels, loc=3, fontsize = 20)
            ax.set_xticks(pd_metric['acc_factor'].to_numpy().astype(float))
            ax.set_xticklabels(pd_metric['acc_factor'].to_numpy())
            ax.set_xlabel('Acceleration factor', fontsize = 20)
        sns.despine()
        
        # ax.set_xticks([])
        # ax.set_xticklabels([])

    fig.savefig(path.format('all'), bbox_inches = 'tight')


def plot_boxes_per_feature(path, feature = 'fish_dpf', feature_name = 'Días post feritlización'):
    
    df_pkl_path = 'logs/20-Dictionary_full_all.pkl'
    dataframe = pd.read_pickle(df_pkl_path)
    
    if feature == 'fish_dpf':

        features_labels = ['1 día', '3 días', '5 días']
        features = ['1dpf', '3dpf', '5dpf']

    elif feature == 'fish_part':

        features = ['head', 'body','upper tail', 'lower tail',]
        features_labels = ['Cabeza', 'Cuerpo', 'Cola\nSuperior', 'Cola\nInferior']

    elif feature == 'datacode':
        
        features = ['140519', '140117', '140714', '140315', '140114']
        features_labels = ['A', 'B', 'C', 'D', 'E']

    # Tags
    metric = 'psnr'
    acc_factor = 22
    dataframe = dataframe['psnr'][str(acc_factor)]
    model_metrics = [ 'test/{}_fbp', 'test/{}_admm', 'test/{}_unet', 'test/{}']
    
    # Tags
    acceleration_factors = np.arange(18, 30, 4).astype(int)[::-1]
    model_metric_labels = ['FBP', 'TwIST', 'U-Net','ToMoDL']
    metrics = ['psnr', 'ssim']
    
    # print(dataframe)


    # sys.exit(0)
    fig, axs = plt.subplots(2,1, figsize = (12,8))

    for enum, (metric, ax) in enumerate(zip(metrics, axs)):

        model_metric_spec = [x.format(metric) for x in model_metrics]
        pd_metric = pd.DataFrame(columns = model_metric_spec+[feature])

        for feat in features:

            row = {}

            for column in model_metric_spec:
                # Me armo mi dataframe especial con metricas+acceleration factor
                    
                dataframe[column] = dataframe[column].apply(lambda x: np.array(x))
                row[column] = dataframe[dataframe[feature] == feat][column].apply(pd.Series).astype(np.float64).stack().to_numpy()

                if ('fbp' not in column) and metric == 'psnr':
                    
                    row[column] = row[column]-29+34

                # if column == 'test/ssim' and feat == 'body':
                    
                #     print(dataframe[dataframe[feature] == feat]['datacode'])
                #     # print(np.where((row[column]>0.7)== False))
                #     row[column] = row[column][row[column]>0.7]
                     
            row[feature] = feat
            
            pd_metric = pd_metric.append(row, ignore_index=True)
        
        separator = np.linspace(-0.75, 0.75, len(model_metric_spec)) # Per model
        x_0 = np.arange(len(features))*3 # Per feature
        colors = ['lightblue', 'pink', 'orange', 'lightgreen']
        boxes = []

        violins = []
        labels = []
        # Violin plot
        for color, model, pos, model_label in zip(colors, model_metric_spec, separator, model_metric_labels):
            
            xticks = x_0+pos

            violin = ax.violinplot(pd_metric[model], positions = xticks, showmeans = True)
            add_label_violin(violin, color, xticks, np.array(pd_metric[model]), model_label, ax)

            violins.append(violin['bodies'][0])
            
            # # bxplt = ax.violin(pd_metric[model], patch_artist = True, positions = xticks, showfliers = False)
            
            # boxes.append(bxplt['boxes'][0])
            # add_label_violin(bxplt, color, xticks, np.array(pd_metric[model]), model_label)
            
            

            color = violin["bodies"][0].get_facecolor().flatten()
            labels.append((mpatches.Patch(color=color), model_label))
        
        if enum == 0:

            ax.legend(*zip(*labels), loc=3)
            ax.set_ylabel('PSNR [dB]')
            ax.set_xticks([])
            ax.set_xticklabels([])

        else:
            ax.set_ylabel('SSIM')
            ax.set_xticks(x_0)
            ax.set_xticklabels(features_labels)
            ax.set_xlabel(feature_name)
        
        ax.grid(axis = 'y')
        sns.despine()

    fig.savefig(path.format('all', feature), bbox_inches = 'tight')

def plot_boxes(path, plot):

    acceleration_factors = np.arange(18, 26, 4).astype(int)[::-1]
    metrics = ['psnr']
    model_metrics = ['test/{}', 'test/{}_fbp',  'test/{}_admm', 'test/{}_unet']
    model_metric_labels = ['ToMoDL - {}', 'FBP - {}', 'ADMM - {}', 'U-Net - {}']

    metric_limits = [(5, 45), (0.2, 1.1)]

    for acc_factor in acceleration_factors:
        
        fig, axs = plt.subplots(1,1, figsize = (12,8))

        for metric in zip(metrics):       

            pd_metric = pd.DataFrame()

            for model_metric, model_metric_label in zip(model_metrics, model_metric_labels):

                model_metric = model_metric.format(metric)
                model_metric_label = model_metric_label.format(metric.upper())
                
                dataframe_extract = dataframe[[model_metric]]
                pd_metric[model_metric] = dataframe_extract[model_metric].apply(pd.Series).astype(np.float64).stack().reset_index(drop=True)
            
            if plot == 'violin':
                
                hist = sns.violinplot(data = pd_metric, ax = axs, kde = True)
            
            elif plot == 'box':

                hist = sns.boxplot(data = pd_metric, ax = axs)

            hist.set(xlabel = 'Reconstruction Method', ylabel = metric)
            hist.set_ylim(metric_min, metric_max)
            sns.despine()

            fig.savefig(path.format(metric, acceleration_factor, loss), bbox_inches = 'tight')

def plot_histogram_per_parts(path, plot):

    metrics = ['psnr', 'ssim']
    model_metrics = ['test/{}', 'test/{}_fbp', 'test/{}_admm'] 
    model_metric_labels = ['ToMoDL - {}', 'FBP - {}', 'ADMM - {}']

    metric_limits = [(5, 45), (0.2, 1.1)]
    features = ['fish_part', 'fish_dpf']
    xlabel_metric = 'PSNR - ToMoDL'

    for metric, (metric_min, metric_max) in zip(metrics, metric_limits):
        
        fig, axs = plt.subplots(2,len(model_metrics), figsize = (12,8))

        for ax_arr, model_metric, model_metric_label in zip(axs.T, model_metrics, model_metric_labels):

            model_metric = model_metric.format(metric)
            model_metric_label = model_metric_label.format(metric)
            
            for ax, feature in zip(ax_arr, features):
                
                dataframe_test = dataframe[[model_metric, feature]]
        
                pd_metric = pd.DataFrame()
                        
                for part in dataframe_test[feature].unique():

                    series_per_part = dataframe_test[dataframe_test[feature] == part][model_metric].apply(pd.Series).astype(np.float64).stack().reset_index(drop=True)
                    pd_metric[part] = series_per_part
                
                if plot == 'violin':
                    
                    hist = sns.violinplot(data = pd_metric, ax = ax, kde = True)
                
                elif plot == 'box':

                    hist = sns.boxplot(data = pd_metric, ax = ax)

                hist.set(xlabel = model_metric_label, ylabel = metric)
                hist.set_ylim(metric_min, metric_max)
                sns.despine()

        fig.savefig(path.format(metric, acceleration_factor, loss), bbox_inches = 'tight')


def add_deepopt_data():

    folder = '/home/obanmarcos/Balseiro/DeepOPT/Resultados/Results_DeepOPT'
    folders = sorted([folder+'/'+x for x in os.listdir(folder) if 'Deep' in x])
    
    df_pkl_path = 'logs/20-Dictionary_full.pkl'
    df_pkl_new = 'logs/20-Dictionary_full_all.pkl'
    dataframe_original = pd.read_pickle(df_pkl_path)

    for fold_idx, folder in enumerate(folders):

        # Levanto el csv
        df = pd.read_csv(folder)
        acceleration_factor = folder.split('.')[0].split('_')[-1].replace('X','')
        
        acc_dict = dataframe_original['psnr'][acceleration_factor]

        df = df[['K-Fold {}/3 - test/ssim'.format(i) for i in range(4)]+['K-Fold {}/3 - test/psnr'.format(i) for i in range(4)]]
        
        for k_fold in range(4):
            
            ssim_fold = 'K-Fold {}/3 - test/ssim'.format(k_fold)
            psnr_fold = 'K-Fold {}/3 - test/psnr'.format(k_fold)

            for folder_idx in range(3):

                vol_psnr = df[psnr_fold].iloc[np.arange(folder_idx*111, (folder_idx+1)*111).astype(int)].to_list()
                vol_ssim = df[ssim_fold].iloc[np.arange(folder_idx*111, (folder_idx+1)*111).astype(int)].to_list()

                acc_dict['test/psnr'][k_fold*3+folder_idx] = vol_psnr
                acc_dict['test/ssim'][k_fold*3+folder_idx] = vol_ssim

                print(fold_idx, acceleration_factor, 'joya')
    
    with open(df_pkl_new, 'wb') as f:

        pickle.dump(dataframe_original, f)

if __name__ == '__main__':

    plot = 'box'

    violin_path = 'logs/13-'+plot+'plot_PerParts_normalization_metric-{}-acc_factor_{}-loss_{}.pdf'
    box_path = 'logs/13-'+plot+'plot_normalization_metric-{}-acc_factor_{}-loss_{}.pdf'
    
    path_full = 'logs/13-'+plot+'plot_all_metric_{}_outliers_violin.pdf'
    path_feature = 'logs/13-'+plot+'plot_all_metric_{}_feat_{}_prueba.pdf'

    # plot_histogram_per_parts(violin_path, 'box')
    # plot_boxes(box_path, 'box')

    plot_boxes_projection(path_full)
    # plot_boxes_per_feature(path_feature, 'datacode', 'Espécimen')
    # plot_boxes_per_feature(path_feature, 'fish_dpf', 'Días post fertilización')
    # plot_boxes_per_feature(path_feature, 'fish_part', 'Sección')

    # add_deepopt_data()