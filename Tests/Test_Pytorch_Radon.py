import os,sys
sys.path.append('/home/marcos/DeepOPT/pytorch_radon')
import pytorch_radon
from pytorch_radon import Radon, IRadon

import phantominator as ph
import torch
import numpy as np
import phantominator as ph
import matplotlib.pyplot as plt

device = torch.device('cuda')
n_angles = 160
image_size = 256
theta = np.linspace(0, 180, n_angles, endpoint=False)
det_count = int(np.sqrt(2)*image_size+0.5)

rad = Radon(image_size, theta, circle=False)
ir = IRadon(image_size, theta, circle = False)

phant = np.copy(np.flipud(ph.shepp_logan(image_size)))
phant_gpu = torch.FloatTensor(phant).to(device)
phant_gpu = phant_gpu[None, None, ...]

sino_gpu = rad(phant_gpu)
backproj = ir(sino_gpu)

fig, ax = plt.subplots(1,3)

ax[0].imshow(phant_gpu.cpu()[0,0,...], cmap = 'gray')
ax[0].set_title('x')
ax[1].imshow(sino_gpu.cpu()[0,0,...], cmap = 'gray')
ax[1].set_title('Ax')
ax[2].imshow(backproj.cpu()[0,0,...], cmap = 'gray')
ax[2].set_title('A^T Ax')

fig.savefig('/home/marcos/DeepOPT/Resultados/Test_PytorchRadon.pdf')
