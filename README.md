# DeepOPT
Model Based Deep Learning Architecture for Optical Tomography Projection 3D reconstruction

### What does this code do:

Based in MoDL architecture developed in [1], this code provides a technique to reconstruct OPT images solving an optimization problem.

As in [[1]], this code solves the optimization problem 

    argmin_x ||Ax-b||_2^2 + ||x-Dw(x)||^2_2

Where 'A' can be any measurement operator, and in our case we considered the Radon transform operator, combined with an undersampling mask. 'Dw(x)' represents the denoiser using a residual learning CNN.

### Dependencies

Radon forward and backward operators used for this code are implemented on Torch Radon library, written by Matteo Ronchetti [2]. It can be installed using:

́́́́`shell script
wget -qO- https://raw.githubusercontent.com/matteo-ronchetti/torch-radon/master/auto_install.py  | python -`

Torch Radon has some compatibility issues with PyTorch above 1.7, so we recommend having a virtual environment with a version between 1.5 and 1.7. It should work with Python 3.8

The training code requires tqdm library in order to track the training progress. 

### OPT datasets

The DeepOPT architecture can be trained using the OPT datasets stored at folder DataOPT. The datasets used along the work correspond to transmitted projection images of live zebrafish at different days post fertilization (dpf). Each dataset consist of a series of 888 sinograms $y \in \mathbb{R}^{D\times\theta}$, sampled at 720 angle steps (0.5 degrees per step) with a detector of 880 pixels, each one evenly spaced by 1.3 $\mu m$.

In order to keep images compatible with Torch Radon operators, sinograms are resized to 640 angle steps so we have a multiple of 16 projections. Image preprocessing involves the registration of opposite pairs to correct the axis of rotation's shift respect to the detector. 


## References
<a id="1">[1]</a>
MoDL: Model Based Deep Learning Architecture for Inverse Problems  by H.K. Aggarwal, M.P Mani, and Mathews Jacob in IEEE Transactions on Medical Imaging, 2018 

<a id="2">[2]</a>
Ronchetti, M. (2020). Torchradon: Fast differentiable routines for computed tomography. arXiv preprint arXiv:2009.14788.