"""
Testing Torch Radon 
author : obanmarcos
"""

from torch_radon import Radon, RadonFanbeam
import torch
from skimage.transform import radon, iradon
import numpy as np
import phantominator as ph
import matplotlib.pyplot as plt

device = torch.device('cuda')
results_folder = '/home/obanmarcos/Balseiro/DeepOPT/Resultados/'

n_angles = 100
image_size = 256
angles = np.linspace(0, np.pi, n_angles, endpoint=False)
det_count = int(np.sqrt(2)*image_size+0.5)

rad = Radon(image_size, angles, clip_to_circle=False, det_count=det_count)

phant = np.copy(np.flipud(ph.shepp_logan(image_size))).astype(float)

phant_gpu = torch.FloatTensor(phant).to(device)
sino_gpu = rad.forward(phant_gpu)/image_size
backproj = rad.backward(sino_gpu)*np.pi/n_angles
filtered_back = rad.backward(rad.filter_sinogram(sino_gpu))

print('x :: Max :', phant_gpu.cpu().max(), 'Min', phant_gpu.cpu().min())
print('Ax :: Max :', sino_gpu.cpu().max(), 'Min', sino_gpu.cpu().min())
print('ATAx :: Max', backproj.cpu().max(), 'Min', backproj.cpu().min())

fig, ax = plt.subplots(1,3)

ax[0].imshow(phant_gpu.cpu(), cmap = 'gray')
ax[0].set_title('x')
ax[1].imshow(sino_gpu.cpu(), cmap = 'gray')
ax[1].set_title('Ax')
ax[2].imshow(backproj.cpu(), cmap = 'gray')
ax[2].set_title('A^T Ax')
fig.savefig(results_folder+'Test_TorchRadon_1.pdf')

fig, ax = plt.subplots(1,2)

ax[0].hist(backproj.cpu())
ax[0].set_title('Sin filtrar')
ax[1].hist(filtered_back.cpu())
ax[1].set_title('Filtrado')
fig.savefig(results_folder+'Test_TorchRadon_Histogram.pdf')
