"""
Test conjugate gradients
"""
import os
import os,time, sys
os.chdir('/home/marcos/DeepOPT/')
sys.path.append('Utilities/')
sys.path.append('OPTmodl/')
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import DataLoading as DL
from Folders_cluster import *
import ModelUtilities as modutils
import torch
from torch_radon.solvers import cg
from torch_radon import Radon, RadonFanbeam
from skimage.transform import radon, iradon
import phantominator as ph
import torchvision
import model_torch as modl
import pickle
from tqdm import tqdm

# Using CPU or GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

img_size = 256
nAngles = 360
lam = 50

image = torch.Tensor(ph.shepp_logan(img_size)).to(device)
image_numpy = ph.shepp_logan(img_size)
A = modl.Aclass(nAngles, img_size, None, lam)

dc = modl.myCG(A, image)
dc_inverse = A.myAtA(dc)

dc_torch_radon = cg(A.myAtA, torch.zeros_like(image), image, max_iter = 10)

# define measuring angles
angles = np.linspace(0, np.pi, nAngles, endpoint=False)

det_count = int(np.sqrt(2)*img_size + 0.5)
print(det_count)

# instantiate Radon transform
radon = Radon(img_size, angles, clip_to_circle=False, det_count=det_count)
hR = lambda x: radon.forward(torch.Tensor(x).to(device)).cpu().numpy()
hRT = lambda x: radon.backprojection(torch.Tensor(x).to(device)).cpu().numpy()

def ConjugateGradient(A, AT, y, delta, max_iter = 5):   
    # % Conjugate gradient routine for linear operators - compressed sensing
    # % A - Forward operator
    # % AT - Backward operator
    # % b - denoised variable ADMM
    # % u - ADMM extra variable for augmented Lagrangian
    # % delta - Regularisation penalty parameter
    # % max_iter - Maximum number of iterations - defaults to 5
    
    xn = np.zeros(y.shape)
    
    rn_1 = y
    pn = rn_1
    k = 0
    rTr = np.dot(rn_1.flatten(), rn_1.flatten())
    
    while (k<max_iter) and (rTr>1e-8):
    
        k += 1
        tn = (AT(A(pn))+delta*pn)    
        alpha = np.dot(rn_1.flatten(), rn_1.flatten())/np.dot(pn.flatten(), tn.flatten())
        
        xn = xn + alpha*pn
        rn_2 = rn_1 - alpha*tn
        
        beta = np.dot(rn_2.flatten(), rn_2.flatten())/np.dot(rn_1.flatten(), rn_1.flatten())
       
        pn = rn_2 + beta*pn   
        rn_1 = rn_2
        rTr = np.dot(rn_1.flatten(), rn_1.flatten())
        print(k, rTr)

    return xn

cg_admm = ConjugateGradient(hR, hRT, image_numpy, lam, 20)

back_admm = hRT(hR(cg_admm))+lam*cg_admm
# direct inversion

#direct_inv = (1/lam)*(image_numpy-(1/(1-lam))*hRT(hR(image_numpy)))


fig, ax = plt.subplots(1,6, figsize = (8,6))

ax[0].imshow(image.cpu().numpy(), cmap = 'gray')
ax[0].set_title('Imagen original')
ax[1].imshow(dc.cpu().numpy(), cmap = 'gray')
ax[1].set_title('CG pytorch')
ax[2].imshow(cg_admm, cmap = 'gray')
ax[2].set_title('ADMM\n Conjugate\n gradients')
ax[3].imshow(dc_torch_radon.cpu().numpy(), cmap = 'gray')
ax[3].set_title('Torch radon CG')
ax[4].imshow(dc_inverse.cpu().numpy(), cmap = 'gray')
ax[4].set_title('Apply back\nAtA+lI\nPytorch')
im = ax[5].imshow(back_admm, cmap = 'gray')
ax[5].set_title('Apply back\nAtA+lI\nADMM')

print((image.cpu().numpy()-dc_inverse.cpu().numpy()).sum())
cax = fig.add_axes([ax[5].get_position().x1+0.01,ax[5].get_position().y0,0.02,ax[5].get_position().height])
plt.colorbar(im, cax = cax)

for a in ax:
    a.axis('off')
fig.savefig(results_folder+'TestConjugateGradients.pdf', bbox_inches = 'tight')
