import os
import os,time, sys
os.chdir('.')
sys.path.append('Utilities/')
sys.path.append('OPTmodl/')
sys.path.append('Reconstruction/')

from Folders_cluster import *
import matplotlib.pyplot as plt
import pickle
import numpy as np
import matplotlib.lines as mlines

with open(results_folder+'Test_PSNR_AllResults.pkl', 'rb') as f:

    loss_tests_PSNR = pickle.load(f)

with open(results_folder+'Test_SSIM_AllResults.pkl', 'rb') as f:
    loss_tests_SSIM = pickle.load(f)

fig, axs = plt.subplots(1,2, figsize = (12,6))

ax2 = axs[1]
ax = axs[0]

for i, ((key, loss_psnr), loss_ssim) in enumerate(zip(loss_tests_PSNR.items(), loss_tests_SSIM.values())):

    loss_psnr = np.array(loss_psnr)
    loss_ssim = np.array(loss_ssim)

    ax.errorbar(i, loss_psnr.mean(), yerr = loss_psnr.std(), fmt = '*', uplims = False, lolims = False, alpha = 0.5, capsize = 5)
    ax2.errorbar(i, loss_ssim.mean(), yerr = loss_ssim.std(), fmt = 'o', uplims = False, lolims = False, alpha = 0.5, capsize = 5)

ax.set_xticks(np.arange(len(loss_tests_PSNR.keys())))
ax.set_xticklabels(loss_tests_PSNR.keys())

ax2.set_xticks(np.arange(len(loss_tests_PSNR.keys())))
ax2.set_xticklabels(loss_tests_PSNR.keys())

ax.set_ylabel('PSNR [dB]')
ax2.set_ylabel('SSIM')
ax.grid(True)
ax2.grid(True)

ssim_marker = mlines.Line2D([], [], color='k', marker='o', linestyle='None', markersize=10, label='SSIM')
psnr_marker = mlines.Line2D([], [], color='k', marker='*', linestyle='None', markersize=10, label='PSNR')
ax.legend(handles=[ssim_marker, psnr_marker])
ax2.legend(handles=[ssim_marker, psnr_marker])

fig.savefig(results_folder+'SSIM_PSNR_Comparison_Separated.pdf', bbox_inches = 'tight')
