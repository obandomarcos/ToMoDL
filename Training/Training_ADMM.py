"""
Test ADMM and comparison
"""
#%% Import libraries
import os
import os, sys
os.chdir('.')
sys.path.append('Utilities/')
sys.path.append('OPTmodl/')
sys.path.append('Reconstruction/')

import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt
import DataLoading as DL
from Folders_cluster import *
import Reconstruction as RecTV
import ModelUtilities as modutils
import torch
from torch_radon import Radon, RadonFanbeam
from skimage.transform import radon, iradon
import torchvision
import model_torch as modl
import pickle
from tqdm import tqdm
from bayes_opt import BayesianOptimization
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.metrics import structural_similarity as ssim
from time import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#%% ADMM test on data
K  = 10 
proj_num = 72
lam = 0.05
nLayer = 8
augment_factor = 1
total_size = 5000
n_angles = 72
max_angle = 640
img_size = 100
det_count = int((img_size+0.5)*np.sqrt(2))
tv_iters = 3
lr = 0.001
shrink = 0.5

train_name_MODL = 'Optimization_Layers_CorrectRegistration_Test52'
train_name_MODLUNET = 'UnetModl_NonRes_Test59'
train_name_UNET = 'Unet_ResVersion_Test58'
train_name_SSIM = 'Optimization_K_SSIM_MSE_Test65'

unet_options = {'residual': False, 'up_conv' :True, 'batch_norm' :True, 'batch_norm_inconv' : True}

model_MODLUNET = modl.OPTmodl(nLayer, K, max_angle, proj_num, img_size, None, lam, results_folder, True,useUnet = 'unet', unet_options = unet_options)
model_UNET = modl.UNet(1,1, residual = True, up_conv = True, batch_norm = True, batch_norm_inconv = True).to(device)
model_MODL = modl.OPTmodl(nLayer, K, max_angle, proj_num, img_size, None, lam, results_folder, True,  useUnet = False)

modutils.load_net(model_folder+train_name_MODL+'K_{}_lam_{}_nlay_{}_proj_{}'.format(K, lam, nLayer, proj_num), model_MODL, device)
modutils.load_net(model_folder+train_name_MODLUNET+'Model_ModlUNet_lr{}_shrink{}'.format(lr, shrink), model_MODLUNET, device)
modutils.load_net(model_folder+train_name_UNET+'Model_Unet_lr{}_shrink{}'.format(lr, shrink), model_UNET, device)

K = 7
nLayer = 8
proj_num = projection = 72
train_factor = 0.7
val_factor = 0.2
test_factor = 0.1
batch_size = 5
lam = 0.001
image_size = 100

model_SSIM = modl.OPTmodl(nLayer, K, max_angle, proj_num, img_size, None, lam, results_folder, True, useUnet = False)
modutils.load_net(model_folder+train_name_SSIM+'K_{}_lam_{}_nlay_{}_proj_{}'.format(K, lam, nLayer, proj_num), model_SSIM, device)

tensor_path = datasets_folder + 'Proj_{}_augmentFactor_{}_totalSize_{}_'.format(proj_num, augment_factor, total_size)                                            
datasets = []
dataloaders = modutils.formDataloaders(datasets, projection, total_size, train_factor, val_factor, test_factor, batch_size, img_size, tensor_path, augment_factor, load_tensor = True, save_tensor = False)    

# Radon operator
n_angles = 720
angles = np.linspace(0, 2*180, n_angles, endpoint = False)
angles_TorchRadon_Full = np.linspace(0, 2*np.pi, n_angles, endpoint = False)
angles_TorchRadon_X10 = np.linspace(0, 2*np.pi, n_angles//10, endpoint = False)

Radon_Full = Radon(image_size, angles_TorchRadon_Full, clip_to_circle=False, det_count=det_count)
Radon_X10 = Radon(image_size, angles_TorchRadon_X10, clip_to_circle=False, det_count=det_count)

Psi = lambda x,th: RecTV.TVdenoise(x,2/th,tv_iters)
#  set the penalty function, to compute the objective
Phi = lambda x: RecTV.TVnorm(x)
hR = lambda x: radon(x, angles, circle = False)
hRT = lambda sino: iradon(sino, angles, circle = False)

#Twist parameters
def obtain_parameters():

    kwargs = {'PSI': Psi, 'PHI':Phi, 'LAMBDA':1e-4, 'TOLERANCEA':1e-4, 'STOPCRITERION': 1, 'VERBOSE': 1, 'INITIALIZATION': 0, 'MAXITERA':10000, 'GPU' : 0}

    loss_test_ADMM = []
    loss_test_fbp = []
    loss_test_modl = []
    loss_test_TWIST = []
    loss_test_MODLUNET = []
    loss_test_UNET = []
    loss_test_SSIM = []

    ssim_ADMM_test = []
    ssim_TWIST_test = []
    ssim_fbp_test = []
    ssim_modl_test = []
    ssim_MODLUNET_test = []
    ssim_UNET_test = []
    ssim_SSIM_test = []

    times_rec_MODL = []
    times_rec_SSIM = []
    times_rec_MODLUNET = []
    times_rec_UNET = []
    times_rec_ADMM = []
    times_rec_TWIST = []
    times_rec_FBPFull =[]
    times_rec_FBPX10 =[]

    num_images = 1000
    shape = int(total_size*test_factor)

    fig_ADMM, ax_ADMM = plt.subplots((shape//num_images)//3+1, 3, figsize = (20, 20))
    fig_FBP, ax_FBP = plt.subplots((shape//num_images)//3+1, 3, figsize = (20, 20))
    fig_MODL, ax_MODL = plt.subplots((shape//num_images)//3+1, 3, figsize = (20, 20))
    fig_TWIST, ax_TWIST = plt.subplots((shape//num_images)//3+1, 3, figsize = (20,20))
    fig_MODLUNET, ax_MODLUNET = plt.subplots((shape//num_images)//3+1, 3, figsize = (20,20))
    fig_UNET, ax_UNET = plt.subplots((shape//num_images)//3+1, 3, figsize = (20,20))
    fig_SSIM, ax_SSIM = plt.subplots((shape//num_images)//3+1, 3, figsize = (20,20))

    ax_ADMM = ax_ADMM.flatten()
    ax_FBP = ax_FBP.flatten()
    ax_MODL = ax_MODL.flatten()
    ax_TWIST = ax_TWIST.flatten()
    ax_MODLUNET = ax_MODLUNET.flatten()
    ax_UNET = ax_UNET.flatten()
    ax_SSIM = ax_SSIM.flatten()

    for a_ADMM, a_FBP, a_MODL, a_TWIST, a_MODLUNET, a_UNET, a_SSIM in zip(ax_ADMM, ax_FBP, ax_MODL, ax_TWIST, ax_MODLUNET, ax_UNET, ax_SSIM):

        a_ADMM.set_axis_off()
        a_FBP.set_axis_off()
        a_MODL.set_axis_off()
        a_TWIST.set_axis_off()
        a_MODLUNET.set_axis_off()
        a_UNET.set_axis_off()
        a_SSIM.set_axis_off()

         
    for i, (imageX_test, imageY_test, imageFiltX_test) in tqdm(enumerate(zip(dataloaders['test']['x'], dataloaders['test']['y'], dataloaders['test']['filtX']))): 

        if (i)%num_images == 0:
            
            sino_full = Radon_Full.forward(imageX_test[0,0,...].to(device))
            sino_X10 = Radon_X10.forward(imageX_test[0,0,...].to(device))
            
            print('Reconstrucción radon')
            time_rec = time() 
            img_rec_FBPFull = Radon_Full.backward(Radon_Full.filter_sinogram(sino_full))
            time_rec = time()-time_rec
            times_rec_FBPFull.append(time_rec)

            print('Reconstrucción radon x10')
            time_rec = time()
            img_rec_FBPFull = Radon_X10.backward(Radon_X10.filter_sinogram(sino_X10))
            time_rec = time()-time_rec
            times_rec_FBPX10.append(time_rec)
            
            print('Reconstrucción MoDL')
            time_rec = time()
            image_rec_MODL = model_MODL(imageX_test.to(device))['dc'+str(K)][0,0,...].detach().cpu().numpy().T  
            time_rec = time()-time_rec
            times_rec_MODL.append(time_rec)
     
            print('Reconstrucción MODL SSIM')
            time_rec = time()
            image_rec_SSIM = model_SSIM(imageX_test.to(device))['dc'+str(K)][0,0,...].detach().cpu().numpy().T   
            time_rec = time()-time_rec
            times_rec_SSIM.append(time_rec)

            print('Reconstrucción MODL UNet')
            time_rec = time()
            image_rec_MODLUNET = model_MODLUNET(imageX_test.to(device))['dc'+str(K)][0,0,...].detach().cpu().numpy().T
            time_rec = time()-time_rec
            times_rec_MODLUNET.append(time_rec)
             
            print('Reconstrucción UNet')
            time_rec = time()
            image_rec_UNET = model_UNET(imageFiltX_test.to(device))[0,0,...].detach().cpu().numpy().T 
            time_rec = time()-time_rec
            times_rec_UNET.append(time_rec)

            image_rec_UNET = (image_rec_UNET-image_rec_UNET.min())/(image_rec_UNET.max()-image_rec_UNET.min())
            image_rec_MODL = (image_rec_MODL-image_rec_MODL.min())/(image_rec_MODL.max()-image_rec_MODL.min())
            image_rec_MODLUNET = (image_rec_MODLUNET-image_rec_MODLUNET.min())/(image_rec_MODLUNET.max()-image_rec_MODLUNET.min())

            imageY_test = imageY_test[0,0,...].to(device).cpu().numpy().T.astype(float)
            imageX_test = imageX_test[0,0,...].to(device).cpu().numpy().T.astype(float)
            imageFiltX_test = imageFiltX_test[0,0,...].to(device).cpu().numpy().T.astype(float)
            
            sino = hR(imageFiltX_test)
            time_rec = time()
            image_rec_ADMM,_,_,_ = RecTV.ADMM(y = sino, A = hR, AT = hRT, Den = Psi, alpha = 0.01,delta = 0.5, max_iter = 10, phi = Phi, tol = 10e-7, invert = 0, warm = 1, true_img = imageY_test)
            time_rec = time()-time_rec
            times_rec_ADMM.append(time_rec)
            image_rec_ADMM = (image_rec_ADMM-image_rec_ADMM.min())/(image_rec_ADMM.max()-image_rec_ADMM.min())
            
            print('Image Y test', imageY_test.min(), imageY_test.max()) 
            print('Image ', image_rec_ADMM.min(), image_rec_ADMM.max())
            
            time_rec = time()
            image_rec_TWIST,_,_,_ = RecTV.TwIST(y = sino, A =hR, AT = hRT, tau = 0.01, kwarg = kwargs , true_img = imageY_test) 
            time_rec = time()-time_rec
            times_rec_TWIST.append(time_rec)

            mse_admm = ((imageY_test - image_rec_ADMM)**2).sum()
            psnr_admm = round(modutils.psnr(img_size, mse_admm, 1), 3)
            ssim_admm = round(ssim(imageY_test,image_rec_ADMM), 3)
            loss_test_ADMM.append(psnr_admm)
            ssim_ADMM_test.append(ssim_admm)

            mse_twist = ((imageY_test - image_rec_TWIST)**2).sum()
            psnr_twist = round(modutils.psnr(img_size, mse_twist, 1), 3)
            ssim_TWIST = round(ssim(imageY_test,image_rec_TWIST), 3)
            loss_test_TWIST.append(psnr_twist)
            ssim_TWIST_test.append(ssim_TWIST)                                   

            mse_fbp = ((imageFiltX_test - imageY_test)**2).sum() 
            psnr_fbp = round(modutils.psnr(img_size, mse_fbp, 1), 3) 
            ssim_FBP = round(ssim(imageY_test,imageFiltX_test), 3)
            loss_test_fbp.append(psnr_fbp)
            ssim_fbp_test.append(ssim_FBP)                                   

            mse_modl = ((image_rec_MODL - imageY_test)**2).sum() 
            psnr_modl = round(modutils.psnr(img_size, mse_modl, 1), 3) 
            ssim_modl = round(ssim(imageY_test,image_rec_MODL), 3)
            loss_test_modl.append(psnr_modl) 
            ssim_modl_test.append(ssim_modl)                                   
            
            mse_MODLUNET = ((image_rec_MODLUNET - imageY_test)**2).sum() 
            psnr_MODLUNET = round(modutils.psnr(img_size, mse_MODLUNET, 1), 3)
            ssim_MODLUNET = round(ssim(imageY_test,image_rec_MODLUNET), 3)
            loss_test_MODLUNET.append(psnr_MODLUNET)
            ssim_MODLUNET_test.append(ssim_MODLUNET)        

            mse_UNET = ((image_rec_UNET - imageY_test)**2).sum() 
            psnr_UNET = round(modutils.psnr(img_size, mse_UNET, 1), 3)  
            ssim_UNET = round(ssim(imageY_test,image_rec_UNET),3)
            loss_test_UNET.append(psnr_UNET)
            ssim_UNET_test.append(ssim_UNET)                                   
            
            print('SSIM entrando')
            mse_SSIM = ((image_rec_SSIM - imageY_test)**2).sum() 
            psnr_SSIM = round(modutils.psnr(img_size, mse_SSIM, 1), 3) 
            ssim_SSIM = round(ssim(imageY_test,image_rec_SSIM),3) 
            loss_test_SSIM.append(psnr_SSIM)                                   
            ssim_SSIM_test.append(ssim_SSIM)                                   

            if i == -1:
                
                print('SSIM done')
                im1 = ax_ADMM[i//num_images].imshow(image_rec_ADMM)
                im2 = ax_FBP[i//num_images].imshow(imageY_test)
                im3 = ax_MODL[i//num_images].imshow(image_rec_MODL)
                im4 = ax_TWIST[i//num_images].imshow(image_rec_TWIST)
                im5 = ax_MODLUNET[i//num_images].imshow(image_rec_MODLUNET)
                im6 = ax_UNET[i//num_images].imshow(image_rec_UNET)
                im7 = ax_SSIM[i//num_images].imshow(image_rec_SSIM)

                divider_ADMM = make_axes_locatable(ax_ADMM[i//num_images])
                divider_FBP = make_axes_locatable(ax_FBP[i//num_images]) 
                divider_MODL = make_axes_locatable(ax_MODL[i//num_images])
                divider_TWIST = make_axes_locatable(ax_TWIST[i//num_images])
                divider_MODLUNET = make_axes_locatable(ax_MODLUNET[i//num_images])
                divider_UNET = make_axes_locatable(ax_UNET[i//num_images])
                divider_SSIM = make_axes_locatable(ax_SSIM[i//num_images])

                cax_ADMM = divider_ADMM.append_axes("right", size="5%", pad=0.05) 
                cax_FBP = divider_FBP.append_axes("right", size="5%", pad=0.05)
                cax_MODL = divider_MODL.append_axes("right", size="5%", pad=0.05)
                cax_TWIST = divider_TWIST.append_axes("right", size="5%", pad=0.05)
                cax_MODLUNET = divider_MODLUNET.append_axes("right", size="5%", pad=0.05)
                cax_UNET = divider_UNET.append_axes("right", size="5%", pad=0.05) 
                cax_SSIM = divider_SSIM.append_axes("right", size="5%", pad=0.05)

                plt.colorbar(im1, cax=cax_ADMM)
                plt.colorbar(im2, cax=cax_FBP)       
                plt.colorbar(im3, cax=cax_MODL)
                plt.colorbar(im4, cax=cax_TWIST) 
                plt.colorbar(im5, cax=cax_MODLUNET)
                plt.colorbar(im6, cax=cax_UNET)
                plt.colorbar(im7, cax=cax_SSIM)
                
                ax_ADMM[i//num_images].set_title('PSNR = {} dB'.format(psnr_admm)+'\nSSIM = {}'.format(ssim_admm))
                ax_FBP[i//num_images].set_title('PSNR = {} dB'.format(psnr_fbp)+'\nSSIM = {}'.format(ssim_FBP)) 
                ax_MODL[i//num_images].set_title('PSNR = {} dB'.format(psnr_modl)+'\nSSIM = {}'.format(ssim_modl))
                ax_TWIST[i//num_images].set_title('PSNR = {} dB'.format(psnr_twist)+'\nSSIM = {}'.format(ssim_TWIST))
                ax_MODLUNET[i//num_images].set_title('PSNR = {} dB'.format(psnr_MODLUNET)+'\nSSIM = {}'.format(ssim_MODLUNET))
                ax_UNET[i//num_images].set_title('PSNR = {} dB'.format(psnr_UNET)+'\nSSIM = {}'.format(ssim_modl))
                ax_SSIM[i//num_images].set_title('PSNR = {} dB'.format(psnr_SSIM)+'\nSSIM = {}'.format(ssim_SSIM))

                fig_ADMM.savefig(results_folder+'ADMMReconstructions.pdf', bbox_inches = 'tight')
                fig_FBP.savefig(results_folder+'FBPReconstructions.pdf', bbox_inches = 'tight')
                fig_MODL.savefig(results_folder+'MODLReconstructions.pdf', bbox_inches = 'tight')
                fig_TWIST.savefig(results_folder+'TWISTReconstructions.pdf', bbox_inches = 'tight')
                fig_MODLUNET.savefig(results_folder+'MODLUNETReconstructions.pdf', bbox_inches = 'tight')
                fig_UNET.savefig(results_folder+'UNETReconstructions.pdf', bbox_inches = 'tight')
                fig_SSIM.savefig(results_folder+'SSIMReconstructions.pdf', bbox_inches = 'tight')
        
            if i== 0:
            
                fig, ax = plt.subplots(2,4, figsize = (20, 10))
                ax = ax.flatten()

                imgs_final = {'ADMM':image_rec_ADMM, 'MODL-UNet': image_rec_MODLUNET, 'MODL\nResNet\nPSNR loss': image_rec_MODL, 'FBP-X10': imageFiltX_test, 'TwIST': image_rec_TWIST, 'U-Net': image_rec_UNET, 'MoDL\nResNet\nSSIM loss': image_rec_SSIM, 'FBP-Full': imageY_test}

                for a, (k, img) in zip(ax, imgs_final.items()):
        
                    print(k)
                    im1 = a.imshow(img, cmap = 'plasma')
                    divider = make_axes_locatable(a)
                    cax = divider.append_axes("right", size="5%", pad=0.05) 
                    plt.colorbar(im1, cax=cax)
                    a.set_xticklabels([])
                    a.set_yticklabels([])
                    
                    if k != 'FBP-Full':
                        mse = ((imageY_test - img)**2).sum()
                        ssim_img = ssim(imageY_test, img)
                        psnr = round(modutils.psnr(img_size, mse, 1), 3)
                    
                    else:      
                        mse = 0
                        ssim_img = 1
                        psnr = r'$\infty$'
                    
                    
                    a.set_title(k+'\nPSNR: {} dB\nSSIM: {}'.format(psnr, round(ssim_img,3)))

                fig.savefig(results_folder+'AllImagesComparison.pdf', bbox_inches = 'tight')
                
                break

def plot_times():

    fig, ax = plt.subplots(1,1, figsize = (8,6))
    #ax = ax.flatten()
    times = {'FBP-Full': np.array(times_rec_FBPFull), 'FBP-X10': np.array(times_rec_FBPX10),'TwIST': np.array(times_rec_TWIST),'ADMM' : np.array(times_rec_ADMM), 'U-Net':np.array(times_rec_UNET), 'MODL\nResNet\nPSNR loss': np.array(times_rec_MODL),'MoDL\nResNet\nSSIM loss': np.array(times_rec_SSIM), 'MODL\nU-Net': np.array(times_rec_MODLUNET)}

    alpha = 0.6
    capsize=5
    elinewidth = 2
    markeredgewidth= 2

    for i, (method, time) in enumerate(times.items()):
        
        ax.errorbar(i, time.mean(), yerr = time.std(), marker = 'h', fmt = '-', alpha = alpha, capsize=capsize, elinewidth=elinewidth, markeredgewidth=markeredgewidth)

    ax.set_yscale('log')
    ax.grid('True')
    ax.set_xticks(np.arange(len(times.items())))
    ax.set_xticklabels(list(times.keys()))

    ax.set_xlabel('Método de reconstrucción')
    ax.set_ylabel('Tiempo de reconstrucción [s]')

    fig.savefig(results_folder+'ReconstructionTimes.pdf', bbox_inches = 'tight')

    print('PSNR:\n---------------\n')
    print('ADMM: {}'.format(np.array(loss_test_ADMM).mean()))
    print('TWIST: {}'.format(np.array(loss_test_TWIST).mean()))
    print('MODL: {}'.format(np.array(loss_test_modl).mean()))
    print('FBP: {}'.format(np.array(loss_test_fbp).mean()))
    print('SSIM as loss: {}'.format(np.array(loss_test_SSIM).mean()))

    print('SSIM:\n---------------\n')
    print('ADMM: {}'.format(np.array(ssim_ADMM_test).mean()))
    print('TWIST: {}'.format(np.array(ssim_TWIST_test).mean()))
    print('MODL: {}'.format(np.array(ssim_modl_test).mean()))
    print('FBP: {}'.format(np.array(ssim_fbp_test).mean()))
    print('SSIM as loss: {}'.format(np.array(ssim_SSIM_test).mean()))

    loss_tests_PSNR = {'FBP-X10' : loss_test_fbp, 'TwIST': loss_test_TWIST, 'ADMM': loss_test_ADMM, 'MODL\nU-Net': loss_test_MODLUNET, 'U-Net' : loss_test_UNET,'MODL\nResNet\nPSNR loss': loss_test_modl, 'MoDL\nResNet\nSSIM loss': loss_test_SSIM}

    with open(results_folder+'Test_PSNR_AllResults.pkl', 'wb') as f:

        pickle.dump(loss_tests_PSNR, f)

    loss_tests_SSIM = {'FBP-X10' : ssim_fbp_test, 'TwIST': ssim_TWIST_test, 'ADMM': ssim_ADMM_test, 'MODL\nU-Net': ssim_MODLUNET_test, 'U-Net' : ssim_UNET_test,'MODL\nResNet\nPSNR loss': ssim_modl_test, 'MoDL\nResNet\nSSIM loss': ssim_SSIM_test}

    with open(results_folder+'Test_SSIM_AllResults.pkl', 'wb') as f:
        pickle.dump(loss_tests_SSIM, f)


def plot_SNR():
    
    alpha = 0.6
    capsize=5
    elinewidth = 2
    markeredgewidth= 2

    with open(results_folder+'Test_PSNR_AllResults.pkl', 'rb') as f:

        loss_tests_PSNR = pickle.load(f)

    with open(results_folder+'Test_SSIM_AllResults.pkl', 'rb') as f:

        loss_tests_SSIM = pickle.load(f)

    errors_psnr_mean = []
    errors_psnr_std = []
    
    errors_ssim_mean = []
    errors_ssim_std = []

    fig, ax = plt.subplots(1,2, figsize = (13,6))

    for i, ((key_PSNR, value_PSNR), (key_SSIM, value_SSIM)) in enumerate(zip(loss_tests_PSNR.items(), loss_tests_SSIM.items())):

        errors_psnr_mean.append(np.array(value_PSNR).mean())
        errors_psnr_std.append(np.array(value_PSNR).std())
        
        errors_ssim_mean.append(np.array(value_SSIM).mean())
        errors_ssim_std.append(np.array(value_SSIM).std())

    ax[0].errorbar(np.arange(len(loss_tests_PSNR.values())), errors_psnr_mean, yerr = errors_psnr_std, marker = 'h', alpha = alpha, capsize=capsize, elinewidth=elinewidth, markeredgewidth=markeredgewidth, linestyle = '')
    ax[1].errorbar(np.arange(len(loss_tests_SSIM.values())), errors_ssim_mean, yerr = errors_ssim_std, c = 'orange', marker = 'h', linestyle = '', alpha = alpha, capsize=capsize, elinewidth=elinewidth, markeredgewidth=markeredgewidth)

    ax[0].set_xticks(np.arange(len(loss_tests_PSNR.values())))
    ax[1].set_xticks(np.arange(len(loss_tests_SSIM.values())))

    ax[0].set_xticklabels(loss_tests_PSNR.keys())
    ax[1].set_xticklabels(loss_tests_SSIM.keys())

    ax[0].grid(True)
    ax[1].grid(True)

    ax[0].set_ylabel('PSNR [dB] on testing images')
    ax[1].set_ylabel('SSIM on testing images')

    fig.savefig(results_folder+'AllMethodsComparison_PSNRSSIM.pdf', bbox_inches = 'tight')


if __name__ == '__main__':

    obtain_parameters()
