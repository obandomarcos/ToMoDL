"""
Testing Torch Radon 
author : obanmarcos
"""

from torch_radon import Radon, RadonFanbeam
import torch
from skimage.transform import radon, iradon
import numpy as np

device = torch.device('cuda')
results_folder = '/home/marcos/DeepOPT/Resultados/'

n_angles = 360
image_size = 512
angles = np.linspace(0, np.pi, n_angles, endpoint=False)
det_count = int(np.sqrt(2)*image_size+0.5)

rad = Radon(image_size, angles, clip_to_circle=False, det_count=det_count)

phant = np.copy(np.flipud(ph.shepp_logan(512)))
phant_gpu = torch.FloatTensor(phant).to(device)
sino_gpu = rad.forward(phant_gpu)

fig, ax = plt.subplots(1,1)

ax[0].imshow(sino_gpu.cpu())
ax[0].set_title('Sinogram GPU')

fig.savefig(results_folder+'Test_TorchRadon_1.pdf')

