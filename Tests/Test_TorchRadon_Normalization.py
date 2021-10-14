import os,sys
os.chdir('/home/marcos/DeepOPT/')
sys.path.append('Utilities/')                                                   
sys.path.append('OPTmodl/')
import numpy as np
from torch_radon import Radon, RadonFanbeam
from torch_radon.solvers import cgne, cg
from Folders_cluster import *
import ModelUtilities as modutils
import matplotlib.pyplot as plt
import torch
import phantominator as ph
from skimage.transform import radon, iradon
import cv2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
img_size = 160
umbral_reg = 50
folder_paths = [f140315_3dpf]

test_name = '/home/marcos/DeepOPT/Tests/Test_sinogram.png'
useSaved = True

# Grab a sinogram of the dataset
if useSaved ==True:
    
    sino_opt = cv2.imread(test_name, cv2.IMREAD_UNCHANGED)
else:
    
    dataset, _ = modutils.formRegDatasets(folder_paths, umbral_reg, img_resize = img_size)
    sino_opt = dataset[0][:,:,200]
    sino_opt = 255*(sino_opt-sino_opt.min())/(sino_opt.max()-sino_opt.min()) 
    cv2.imwrite(test_name, img_opt)

#print('Sinogram shape', sinogram.shape)
print('sinogram OPT shape', sino_opt.shape)

# Generate phantom
#img = ph.shepp_logan(img_size)

n_angles = 720
angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
det_count = int((np.sqrt(2)*img_size)+0.5)

rad = Radon(img_size, angles, clip_to_circle = False, det_count = det_count)
#img_radon_cpu = iradon(img_opt, angles*180/np.pi) 

sino_opt = torch.FloatTensor(sino_opt).to(device)
sino_opt = (sino_opt-sino_opt.min())/(sino_opt.max()-sino_opt.min())
print('sinogram values', sino_opt.max(), sino_opt.min())

img_opt = rad.backward(sino_opt)/(n_angles)
print('image values', img_opt.max(), img_opt.min())

img_opt_filt = rad.backward(rad.filter_sinogram(sino_opt))
print('filtered values', img_opt_filt.max(), img_opt_filt.min())

fig, ax = plt.subplots(1,3)
ax[0].imshow(sino_opt.cpu(), cmap = 'gray')
ax[0].set_title('Sinograma')
ax[1].imshow(img_opt.cpu(), cmap = 'gray')
ax[1].set_title('Reconstrucción\nTorch Radon\nNo Filter')
ax[2].imshow(img_opt_filt.cpu() , cmap = 'gray')
ax[2].set_title('Reconstrucción\n Radon\n Filter')
fig.savefig(results_folder+'Img_Sinogram_test.pdf', bbox_inches = 'tight')
