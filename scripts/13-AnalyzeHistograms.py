import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import numpy as np

df_pkl_path = 'logs/test_dataframe_x22.pkl'
dataframe_x22 = pd.read_pickle(df_pkl_path)

# Subset by part

metrics = ['psnr', 'ssim']
model_metrics = ['test/{}', 'test/{}_fbp'] 
model_metric_labels = ['MoDL - {}', 'FBP - {}']

metric_limits = [(5, 45), (0.2, 1.1)]
features = ['fish_part', 'fish_dpf']
xlabel_metric = 'PSNR - MoDL'

for metric, (metric_min, metric_max) in zip(metrics, metric_limits):
    
    fig, axs = plt.subplots(2,2, figsize = (8,6))

    for ax_arr, model_metric, model_metric_label in zip(axs.T, model_metrics, model_metric_labels):

        model_metric = model_metric.format(metric)
        model_metric_label = model_metric_label.format(metric)
        
        for ax, feature in zip(ax_arr, features):
            
            dataframe_test = dataframe_x22[[model_metric, feature]]
    
            pd_metric = pd.DataFrame()
                    
            for part in dataframe_test[feature].unique():

                series_per_part = dataframe_test[dataframe_test[feature] == part][model_metric].apply(pd.Series).astype(np.float64).stack().reset_index(drop=True)
                pd_metric[part] = series_per_part
                
            hist = sns.violinplot(data = pd_metric, ax = ax, alpha = 0.5, kde = True)
            hist.set(xlabel = model_metric_label, ylabel = metric)
            hist.set_ylim(metric_min, metric_max)
            sns.despine()

    fig.savefig('logs/13-Violin_metric-{}.pdf'.format(metric), bbox_inches = 'tight')