import os,sys
os.chdir('/home/marcos/DeepOPT/')
sys.path.append('Utilities/')                                                   
sys.path.append('OPTmodl/')
import numpy as np
from torch_radon import Radon, RadonFanbeam
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

test_name = '/home/marcos/DeepOPT/Tests/Test_sinogram.jpg'
useSaved = False

# Grab a sinogram of the dataset
if useSaved ==True:
    
    img = cv2.imread(test_name)
    
else:
    
    dataset, _ = modutils.formRegDatasets(folder_paths, umbral_reg, img_resize = img_size)
    img = dataset[0][:,:,200].T
    
    cv2.imwrite(test_name, img)

#img = ph.shepp_logan(100)

n_angles = 720
angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
det_count = int((np.sqrt(2)*img_size)+0.5)

radon = Radon(img_size, angles, clip_to_circle = True, det_count = det_count)

img_radon_cpu = iradon(img, angles*180/np.pi) 
img = torch.FloatTensor(img).to(device)
#img = radon.forward(img)

fig, ax = plt.subplots(1,3)
ax[0].imshow(img.cpu(), cmap = 'gray')
ax[0].set_title('Sinograma')
ax[1].imshow(radon.backward(radon.filter_sinogram(img)).cpu(), cmap = 'gray')
ax[1].set_title('Reconstrucción\nTorch Radon')
ax[2].imshow(img_radon_cpu , cmap = 'gray')
ax[2].set_title('Reconstrucción\nScikit')
fig.savefig(results_folder+'Img_Sinogram_test.pdf', bbox_inches = 'tight')

