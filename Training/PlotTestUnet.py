'''
Plot Test results for Unet
'''
import pickle
import os, sys
os.chdir('/home/marcos/DeepOPT/')
sys.path.append('Utilities/')
sys.path.append('OPTmodl/')
import matplotlib.pyplot as plt
import numpy as np
from Folders_cluster import *
import ModelUtilities as modutils

max_angle = 640
shrink = 0.5
lr = 0.001 
train_size = 0.7
train_name_unet = 'Unet_CorrectRegistration_Test54'
batch_size = 5
train_name_modlunet = 'UnetModl_CorrectRegistration_Test53'

with open(results_folder+train_name_unet+'Train_UNet_lr{}_shrink{}.pkl'.format(lr, shrink), 'rb') as f:

    training_full_unet = pickle.load(f)

with open(results_folder+train_name_modlunet+'Train_ModlUNet_lr{}_shrink{}.pkl'.format(lr, shrink), 'rb') as f: 

    training_full_modlunet = pickle.load(f)


print(training_full_modlunet)
Unet_means = []
fbp_loss = []

#Unet_means.append(np.mean(np.array(test_loss_total['loss_net'])))
#fbp_loss.append(np.mean(np.array(test_loss_total['loss_fbp'])))

print(Unet_means, fbp_loss)

fig, ax = plt.subplots(1,2, figsize = (16, 8), sharey = True)

ax[0].plot(training_full_unet['train'], label = 'Unet Train')
ax[0].plot(training_full_unet['train_fbp'], label = 'FBP')
ax[0].plot(training_full_unet['val'], label = 'Unet Validation')
ax[0].set_title('Unet raw')
ax[0].grid()
ax[0].set_ylabel('Absolute error (L1)')
ax[0].legend()

ax[1].set_title('Unet+Modl')
ax[1].plot(training_full_modlunet['train'], label = 'MODL Unet Train')
ax[1].plot(training_full_modlunet['val'], label = 'MODL Unet Val')
ax[1].plot(training_full_modlunet['train_fbp'], label = 'FBP')
ax[1].grid()
ax[1].set_xlabel('Épocas')
ax[1].legend()
fig.savefig(results_folder+'Unet_Train.pdf', bbox_inches = 'tight')

with open(results_folder+train_name_modlunet+'Test_UnetMoDL_lr{}_shrink{}.pkl'.format(lr, shrink), 'rb') as f:
    
    unpickle = pickle.Unpickler(f) 
    test_loss_modlUnet = unpickle.load()
    print('Diccionario cargado para proyección {}, MODL+UNET')

with open(results_folder+train_name_unet+'Test_Unet_lr{}_shrink{}.pkl'.format(lr, shrink), 'rb') as f:
    
    unpickle = pickle.Unpickler(f)
    test_loss_unet = unpickle.load()
    print('Diccionario cargado para UNET')

print('PSNR mean Test\n MODL : {} dB\n UNET : {} dB\n FBP : {}'.format(round(np.array(test_loss_modlUnet['loss_net']).mean(), 2), round(np.array(test_loss_unet['loss_net']).mean(),2),round(np.array(test_loss_unet['loss_fbp']).mean(),2)))

