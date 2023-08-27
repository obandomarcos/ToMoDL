'''
Training samples analysis

'''

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


from matplotlib import rcParams, cycler
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['font.size'] = 12
rcParams['axes.linewidth'] = 1.1
rcParams['axes.labelpad'] = 10.0
plot_color_cycle = cycler('color', ['000000', '0000FE', 'FE0000', '008001', 'FD8000', '8c564b', 
                                    'e377c2', '7f7f7f', 'bcbd22', '17becf'])
rcParams['axes.prop_cycle'] = plot_color_cycle
rcParams['axes.xmargin'] = 0
rcParams['axes.ymargin'] = 0
rcParams.update({"figure.figsize" : (6.4,4.8),
                 "figure.subplot.left" : 0.177, "figure.subplot.right" : 0.946,
                 "figure.subplot.bottom" : 0.156, "figure.subplot.top" : 0.965,
                 "axes.autolimit_mode" : "round_numbers",
                 "xtick.major.size"     : 7,
                 "xtick.minor.size"     : 3.5,
                 "xtick.major.width"    : 1.1,
                 "xtick.minor.width"    : 1.1,
                 "xtick.major.pad"      : 5,
                 "xtick.minor.visible" : True,
                 "ytick.major.size"     : 7,
                 "ytick.minor.size"     : 3.5,
                 "ytick.major.width"    : 1.1,
                 "ytick.minor.width"    : 1.1,
                 "ytick.major.pad"      : 5,
                 "ytick.minor.visible" : True,
                 "lines.markersize" : 10,
                 "lines.markerfacecolor" : "none",
                 "lines.markeredgewidth"  : 0.8})

results_tomodl_ssim_path = '/home/obanmarcos/Balseiro/DeepOPT/results/training_samples_tomodl_ssim.csv'
results_tomodl_psnr_path = '/home/obanmarcos/Balseiro/DeepOPT/results/training_samples_tomodl_psnr.csv'

results_unet_ssim_path = '/home/obanmarcos/Balseiro/DeepOPT/results/training_samples_unet_ssim.csv'
results_unet_psnr_path = '/home/obanmarcos/Balseiro/DeepOPT/results/training_samples_unet_psnr.csv'

results_tomodl_df_ssim = pd.read_csv(results_tomodl_ssim_path)
results_tomodl_df_psnr = pd.read_csv(results_tomodl_psnr_path)
results_unet_df_ssim = pd.read_csv(results_unet_ssim_path)
results_unet_df_psnr = pd.read_csv(results_unet_psnr_path)

results_unet_df_psnr['test/psnr'] = results_unet_df_psnr['test/psnr']

dfs_tomodl = [results_tomodl_df_ssim, results_tomodl_df_psnr]
dfs_unet = [results_unet_df_ssim, results_unet_df_psnr]

for df in dfs_tomodl:
    
    df['Name'].replace({'Group: Duggan': 1, 'Group: Flora': 2, 'Group: Nguyen': 5, 'Group: Driscoll': 10, 'Group: Griffin': 20, 'Group: Caldwell':50, 'Group: Lyles':100 }, inplace=True)
    df.sort_values('Name', inplace=True)
    df.rename(columns={'test/ssim':'test/ssim_tomodl', 'test/psnr':'test/psnr_tomodl'})

for df in dfs_unet:
    
    df['Name'].replace({'Group: Kaupp': 1, 'Group: Machak': 2, 'Group: Nixon': 5, 'Group: Burton': 10, 'Group: Winkleman': 20, 'Group: Watson': 50, 'Group: Florido': 100}, inplace=True)
    df.rename(columns={'test/ssim':'test/ssim_unet', 'test/psnr':'test/psnr_unet'})
    df.sort_values('Name', inplace=True)

print(dfs_unet[0])
# results_tomodl_df['Name'] = results_tomodl_df['Name'].str.replace('%', '').astype(float)

w = 0.1
width = lambda p, w: 10**(np.log10(p)+w/2.)-10**(np.log10(p)-w/2.)
positions =np.array([1,2,5,10,20,50])
off_box = 0.5

fig, axs = plt.subplots(2,1, figsize = (8, 10), sharex=True)

for enum, (ax, y_label, y_title, df_tomodl, df_unet, off) in enumerate(zip(axs, ['test/ssim', 'test/psnr'], ['SSIM', 'PSNR [dB]'], dfs_tomodl, dfs_unet, [0.002, 0.1])):
    # Customize the plot

    ax.set_xscale('log')
    # axes = df_tomodl.boxplot(column = y_label, by = 'Name', ax = ax,  widths = width(positions, w), positions = positions, showfliers = False,color = 'blue', showbox = False)

    # axes = df_unet.boxplot(column = y_label, by = 'Name', ax = ax, widths = width(positions, w), positions = positions, showfliers = False, color = 'green', showbox = False)

    if enum == 0:
        axes = df_unet.groupby(['Name']).mean()[:-1].plot(y = y_label, yerr = df_unet.groupby(['Name']).std(), marker='.', linestyle='None', ax = ax, color = 'green',capsize=5)
        axes = df_tomodl.groupby(['Name']).mean()[:-1].plot(y = y_label,  yerr = df_unet.groupby(['Name']).std(), marker='.', linestyle='None', ax = ax, color = 'blue', capsize=5)

    else:
        axes = df_unet.groupby(['Name']).mean()[:-1].plot(y = y_label, yerr = df_unet.groupby(['Name']).std(), marker='.', linestyle='None', ax = ax, color = 'green',capsize=5, legend=False)
        axes = df_tomodl.groupby(['Name']).mean()[:-1].plot(y = y_label,  yerr = df_unet.groupby(['Name']).std(), marker='.', linestyle='None', ax = ax, color = 'blue', capsize=5, legend=False)
        
    ax.set_title('')

    ax.set_xticks(positions)
    ax.set_xticklabels(['1%','2%','5%','10%','20%', '50%'])

    # ax.semilogx([1,2,5,10,20,50])
    # ax.text(2.7, df[df['Name']=='1%'][y_label].astype(float).mean()+off,'FBP \nreconstruction', fontdict={'fontsize':8})

    ax.grid(True, alpha = 0.3)
    ax.set_ylabel(y_title)
    ax.set_xlabel('')
    ax.axhline(df_unet[df_unet['Name']==1][y_label].astype(float).mean(), color='C0', linestyle = '--', label = 'FBP\nReconstruction')
    
    handles, labels = ax.get_legend_handles_labels()

    labels = labels[:1]+['U-Net', 'ToMoDL']

    if enum == 0:
        leg = ax.legend(handles, labels)
        for lh in leg.legendHandles: 
            lh.set_alpha(1)

    ax.set_xlim(0.8, 80)
    
ax.set_xlabel('Training Samples')
fig.suptitle('')
# Show the plot

fig.savefig('/home/obanmarcos/Balseiro/DeepOPT/results/26-TrainingViolin.pdf', bbox_inches = 'tight')