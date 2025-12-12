"""
Deep-learning–enhanced tomographic reconstruction framework.

This module implements classical and learned reconstruction pipelines for
Optical Projection Tomography (OPT) and X-ray CT, including:

• CPU/GPU filtered backprojection (FBP)
• TwIST and Total Variation–based iterative reconstruction
• MoDL / ToMoDL model-based deep learning reconstruction
• U-Net and ResNet-style learned denoisers
• Conjugate-gradient–based data consistency (AtA) operator
• Utilities for filtering, normalization, and sinogram handling

Backends supported:
    – scikit-image Radon/iradon
    – QBI-Radon (PyTorch, GPU-accelerated)
    – Custom UNet and ResNet denoisers

The module integrates classical tomography algorithms with deep unfolded
optimization models, following methods such as MoDL (Aggarwal et al. 2018) and
deep OPT reconstruction (Davis et al., 2019).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from skimage.transform import radon as radon_scikit
from skimage.transform import iradon as iradon_scikit
from skimage.filters import median
from skimage.transform import resize as resize_skimage
from skimage.morphology import disk
from .unet import UNet

from .alternating import TwIST, TVdenoise, TVnorm
import numpy as np
import os
# import cv2
from enum import Enum
import tqdm
import os
try:
    from QBI_radon import Radon as radon_thrad
    from QBI_radon import ramp_filter_torch

    use_torch_radon = True
    use_tomopy = False
    use_scikit = False

    print("QBI-Radon available!")

except:

    print("QBI-Radon not available!")
    use_torch_radon = False
    use_tomopy = False
    use_scikit = True

from skimage.transform import radon, iradon
import numpy as np
from . import unet


# Modify for multi-gpu
device = torch.device("cuda:0" if use_torch_radon == True else "cpu")

def mean_std_median_filter(images):
    """Compute mean and standard deviation after median filtering.

    Applies a median filter (disk radius 5) to each image and computes the
    filtered mean and standard deviation. Useful for intensity renormalization
    in learned reconstruction.

    Args:
        images (ndarray): Array of images with shape (N, 1, H, W) or (N, 1, H).

    Returns:
        tuple(ndarray, ndarray):
            mean_images: Mean intensity per image, shape (N, 1)
            std_images: Standard deviation per image, shape (N, 1)
    """
    mean_images = np.zeros((images.shape[0], 1))
    std_images = np.zeros((images.shape[0], 1))
    for i, image in enumerate(images):
        image_median = median(image[0], disk(5))
        mean_images[i, 0] = image_median.mean()
        std_images[i, 0] = image_median.std()
    return mean_images, std_images

def ramp_filter(sinogram):
    """Apply a discrete ramp filter to a sinogram.

    Implements the Ram-Lak (ramp) filter using FFT with zero-padding.

    Args:
        sinogram (ndarray): Sinogram of shape (num_detectors, num_angles).

    Returns:
        ndarray: Filtered sinogram with the same shape.

    Notes:
        This is a CPU, NumPy-only implementation.
    """
    num_detectors, num_angles = sinogram.shape
    n = int(2 ** np.ceil(np.log2(num_detectors)))  # zero-padding for FFT
    freqs = np.fft.fftfreq(n).reshape(-1, 1)
    ramp = 2 * np.abs(freqs)

    filtered_sino = np.zeros_like(sinogram)

    for i in range(num_angles):
        projection = sinogram[:, i]
        padded = np.zeros(n)
        padded[:num_detectors] = projection

        proj_fft = np.fft.fft(padded)
        filtered_fft = proj_fft * ramp[:, 0]
        filtered_proj = np.real(np.fft.ifft(filtered_fft))[:num_detectors]

        filtered_sino[:, i] = filtered_proj

    return filtered_sino

#     return filtered_sino
class dwLayer(nn.Module):
    """Single convolutional layer used in the learned denoiser (dw block).

    Supports:
        • Optional batch normalization
        • Xavier / constant initialization
        • Swish (SiLU) activation unless last layer

    Args:
        kw_dictionary (dict): Contains:
            weights_size (tuple): (in_channels, out_channels, kernel, stride)
            is_last_layer (bool): If True, skip activation
            init_method (str): "xavier" or "constant"
            use_batch_norm (bool): Whether to apply batch normalization
    """

    def __init__(self, kw_dictionary):
        """
        Dw component initializer
        Params:
            - weights_size (tuple): convolutional neural network size (in_channels, out_channels, kernel_size)
            - is_last_layer (bool): if True, Relu is not applied
            - init_method (string): Initialization method, defaults to Xavier init
        """

        super().__init__()

        self.process_kwdictionary(kw_dictionary)
        self.conv = nn.Conv2d(
            *self.weights_size, padding=(int(self.weights_size[2] / 2), int(self.weights_size[2] / 2))
        )

        self.initialize_layer(method=self.init_method)

        if self.use_batch_norm == True:
            self.batch_norm = nn.BatchNorm2d(self.weights_size[1])

    def forward(self, x):
        """
        Forward pass for block
        Params:
            - x (torch.Tensor): Image batch to be processed
        """
        output = self.conv(x)

        if self.use_batch_norm:

            output = self.batch_norm(output)

        if self.is_last_layer != True:

            # output = F.relu(output)
            # use swish activation function
            output = F.silu(output)

        return output

    def process_kwdictionary(self, kw_dictionary):
        """
        Process keyword dictionary.
        Params:
            - kw_dictionary (dict): Dictionary with keywords
        """

        self.weights_size = kw_dictionary["weights_size"]
        self.is_last_layer = kw_dictionary["is_last_layer"]
        self.init_method = kw_dictionary["init_method"]
        self.use_batch_norm = kw_dictionary["use_batch_norm"]

    def initialize_layer(self, method):
        """
        Initializes convolutional weights according to method
        Params:
         - method (string): Method of initialization, please refer to https://pytorch.org/docs/stable/nn.init.html
        """
        if method == "xavier":
            return
        elif method == "constant":
            torch.nn.init.constant_(self.conv.weight, 0.001)
            torch.nn.init.constant_(self.conv.bias, 0.001)


class dw(nn.Module):
    """Residual convolutional denoiser block.

    Builds a stack of convolutional layers followed by a residual connection:
        output = F(x) + x

    Args:
        kw_dictionary (dict): Contains:
            number_layers (int)
            kernel_size (int)
            features (int)
            in_channels (int)
            out_channels (int)
            stride (int)
            use_batch_norm (bool)
            init_method (str)
            device (torch.device)

    Forward:
        x (Tensor): Input batch (B, C, H, W)
        Returns denoised output.
    """
    def __init__(self, kw_dictionary):
        """
        Initialises dw block
        Params:
            - kw_dictionary (dict): Parameters dictionary
        """
        super(dw, self).__init__()
        self.process_kwdictionary(kw_dictionary=kw_dictionary)

        for i in np.arange(1, self.number_layers + 1):

            self.dw_layer_dict["weights_size"] = self.weights_size[i]

            if i == self.number_layers - 1:
                self.dw_layer_dict["is_last_layer"] = True

            self.nw["c" + str(i)] = dwLayer(self.dw_layer_dict).to(self.device)

        self.nw = nn.ModuleDict(self.nw)

    def forward(self, x):
        """
        Forward pass
        Params:
            - x (torch.Tensor): Image batch to be processed
        """
        residual = torch.clone(x)

        for layer in self.nw.values():

            x = layer(x)

        output = x + residual

        return output

    def process_kwdictionary(self, kw_dictionary):
        """
        Process keyword dictionary.
        Params:
            - kw_dictionary (dict): Dictionary with keywords
        """

        self.number_layers = kw_dictionary["number_layers"]
        self.nw = {}
        self.kernel_size = kw_dictionary["kernel_size"]
        self.features = kw_dictionary["features"]
        self.in_channels = kw_dictionary["in_channels"]
        self.out_channels = kw_dictionary["out_channels"]
        self.stride = kw_dictionary["stride"]
        self.use_batch_norm = kw_dictionary["use_batch_norm"]
        self.init_method = kw_dictionary["init_method"]
        self.device = kw_dictionary["device"]

        # Intermediate layers (in_channels, out_channels, kernel_size_x, kernel_size_y)
        self.weights_size = {
            key: (self.features, self.features, self.kernel_size, self.stride) for key in range(2, self.number_layers)
        }
        self.weights_size[1] = (self.in_channels, self.features, self.kernel_size, self.stride)
        self.weights_size[self.number_layers] = (self.features, self.out_channels, self.kernel_size, self.stride)

        self.dw_layer_dict = {
            "use_batch_norm": self.use_batch_norm,
            "is_last_layer": False,
            "init_method": self.init_method,
        }


class Aclass:
    """Data-consistency (DC) operator for MoDL / ToMoDL reconstruction.

    Implements the operator:
        (AᵀA + λI)x
    where A is the Radon transform.

    Supports:
        • QBI-Radon GPU implementation
        • scikit-image CPU implementation
        • Conjugate gradient inversion

    Args:
        kw_dictionary (dict):
            image_size (int)
            number_projections (int)
            lambda (float)
            use_torch_radon (bool)
            use_scikit (bool)
            iter_conjugate (int)
            device (torch.device)
            is_half_rotation (bool)
    """

    def __init__(self, kw_dictionary):
        """
        Initializes Conjugate gradients step.
        Params:
            - kw_dictionary (dict): Keyword dictionary
        """

        self.img_size = kw_dictionary["image_size"]
        self.number_projections = kw_dictionary["number_projections"]
        self.lam = kw_dictionary["lambda"]
        self.use_torch_radon = kw_dictionary["use_torch_radon"]
        self.use_scikit = kw_dictionary["use_scikit"]

        self.is_half_rotation = kw_dictionary["is_half_rotation"]
        if self.is_half_rotation == True:
            self.angles = np.linspace(0, np.pi, self.number_projections, endpoint=False)
        else:
            self.angles = np.linspace(0, 2 * np.pi, self.number_projections, endpoint=False)
        self.det_count = int(np.ceil(np.sqrt(2) * self.img_size))
        self.device = kw_dictionary["device"]
        self.iter_conjugate = kw_dictionary["iter_conjugate"]
        if self.use_torch_radon == True:
            self.radon = radon_thrad(thetas=self.angles, circle=False, device=self.device, filter_name=None)

        elif self.use_scikit == True:

            class Radon:
                def __init__(self, num_angles, circle=True):
                    self.num_angles = num_angles
                    self.circle = circle

                def forward(self, image):
                    # Compute the Radon transform of the image
                    image = image.detach().numpy()
                    sinogram = radon(image, theta=np.linspace(0, 2 * 180, self.num_angles), circle=self.circle)
                    sinogram = torch.tensor(sinogram).to(self.device)
                    return sinogram

                def backprojection(self, sinogram):
                    # Compute the backprojection of the sinogram
                    sinogram = sinogram.detach().numpy()
                    reconstruction = iradon(
                        sinogram, theta=np.linspace(0, 2 * 180, self.num_angles), circle=self.circle, filter_name=None
                    )
                    reconstruction = torch.tensor(reconstruction).to(self.device)
                    return reconstruction

            self.radon = Radon(self.number_projections, circle=False)

    def forward(self, img):
        """Apply (AᵀA + λI) to an image tensor."""

        # sinogram = self.radon.forward(img)/self.img_size
        # iradon = self.radon.backprojection(sinogram)*np.pi/self.number_projections
        # del sinogram
        # output = iradon+self.lam*img

        sinogram = self.radon(img) / self.img_size
        iradon = self.radon.filter_backprojection(sinogram) * np.pi / self.number_projections
        output = iradon + self.lam * img

        # print('output forward: {} {}'.format(output.max(), output.min()))
        # print('Term z max {}, min {}'.format((iradon/self.lam).max(), (iradon/self.lam).min()))
        # print('Term input max {}, min {}'.format(img.max(), img.min()))
        # print('Term output max {}, min {}'.format(output.max(), output.min()))
        return output

    def inverse(self, rhs):
        """Solve (AᵀA + λI)x = rhs using conjugate gradients.

        Args:
            rhs (Tensor): Right-hand side tensor.

        Returns:
            Tensor: Approximate solution of the linear system.
        """
        y = self.conjugate_gradients(self.forward, rhs)  # This indexing may fail

        return y

    # @staticmethod
    def conjugate_gradients(self, A, rhs):
        """Conjugate gradient solver in PyTorch.

        Solves:
            minimize_x ||Ax – rhs||²

        Args:
            A (callable): Linear operator.
            rhs (Tensor): Right-hand side vector.

        Returns:
            Tensor: Estimated solution.
        """

        i = 0
        x = torch.zeros_like(rhs)
        r = rhs
        p = rhs
        rTr = torch.sum(r * r)

        while (i < self.iter_conjugate) and torch.ge(rTr, 1e-5):

            # print(rTr)
            Ap = A(p)
            alpha = rTr / torch.sum(p * Ap)
            x = x + alpha * p
            r = r - alpha * Ap
            rTrNew = torch.sum(r * r)
            beta = rTrNew / rTr
            p = r + beta * p
            i += 1
            rTr = rTrNew

        # print('output CG: {} {}'.format(x.max(), x.min()))
        return x


class ToMoDL(nn.Module):
    """Deep unrolled model for tomographic reconstruction (MoDL-style).

    Combines:
        • Learned denoiser (U-Net or ResNet)
        • Data-consistency step (CG inversion)
        • K unrolled iterations

    Based on:
        - Aggarwal et al., "MoDL: Model-based deep learning...", TMI 2018.
        - Davis et al., "CNNs for undersampled OPT reconstruction", 2019.

    Args:
        kw_dictionary (dict):
            K_iterations (int): Number of unrolled iterations
            lambda (float)
            image_size (int)
            number_projections_total (int)
            acceleration_factor (int)
            denoiser_method ("U-Net" or "resnet")
            device (torch.device)
    """
    def __init__(self, kw_dictionary):
        """
        Main function that creates the model
        Params :

            - nLayer (int): Number of layers
            - K (int): unrolled network number of iterations
            - n_angles (int): Number of total angles of the sinogram, fully sampled
            - proj_num (int): Number of undersampled angles of the model
            - image_size (int): Image size in pixels
            -

        """
        super(ToMoDL, self).__init__()

        self.process_kwdictionary(kw_dictionary)
        self.define_denoiser()

    def forward(self, x):
        """Perform K unrolled DC–denoise iterations.

        Args:
            x (Tensor): Backprojected initial reconstruction (B, 1, H, W).

        Returns:
            dict: Outputs of each iteration (dc0, dw1, dc1, ..., dcK).
        """

        self.out["dc0"] = x

        # for i in range(1, self.K + 1):

        #     j = str(i)

        #     self.out["dw" + j] = normalize_images(self.dw.forward(self.out["dc" + str(i - 1)]))
        #     rhs = self.out["dc0"] / self.lam + self.out["dw" + j]

        #     self.out["dc" + j] = normalize_images(self.AtA.inverse(rhs))

        #     del rhs
        #####################################################################################
        for i in range(1, self.K + 1):
            j = str(i)
            self.out["dw" + j] = self.dw.forward(self.out["dc" + str(i - 1)])
            rhs = x / self.lam + self.out["dw" + j]

            self.out["dc" + j] = self.AtA.inverse(rhs)
            del rhs

        return self.out

    def process_kwdictionary(self, kw_dictionary):
        """
        Process keyword dictionary.
        Params:
            - kw_dictionary (dict): Dictionary with keywords
        """

        self.out = {}
        self.use_torch_radon = use_torch_radon
        self.use_scikit = use_scikit
        self.use_tomopy = use_tomopy

        self.device = kw_dictionary["device"]
        self.K = kw_dictionary["K_iterations"]
        self.number_projections_total = kw_dictionary["number_projections_total"]
        self.acceleration_factor = kw_dictionary["acceleration_factor"]
        self.number_projections_undersampled = self.number_projections_total // self.acceleration_factor
        self.image_size = kw_dictionary["image_size"]

        self.lam = kw_dictionary["lambda"]
        self.lam = torch.nn.Parameter(torch.tensor([self.lam], requires_grad=False, device=self.device))

        self.use_shared_weights = kw_dictionary["use_shared_weights"]
        self.denoiser_method = kw_dictionary["denoiser_method"]

        self.in_channels = kw_dictionary["in_channels"]
        self.out_channels = kw_dictionary["out_channels"]
        self.iter_conjugate = kw_dictionary["iter_conjugate"]
        if self.denoiser_method == "U-Net":
            self.unet_options = kw_dictionary["unet_options"]
        elif self.denoiser_method == "resnet":
            self.resnet_options = kw_dictionary["resnet_options"]
        self.AtA_dictionary = {
            "image_size": self.image_size,
            "number_projections": self.number_projections_total,
            "lambda": self.lam,
            "use_torch_radon": self.use_torch_radon,
            "use_scikit": self.use_scikit,
            "use_tomopy": self.use_tomopy,
            "device": self.device,
            "iter_conjugate": self.iter_conjugate,
            "is_half_rotation": kw_dictionary["is_half_rotation"],
        }

        self.AtA = Aclass(self.AtA_dictionary)

    def define_denoiser(self):
        """
        Defines denoiser used in MoDL. Options include Resnet and U-Net

        References:
            - Aggarwal, H. K., Mani, M. P., & Jacob, M. (2018). MoDL: Model-based deep learning architecture for inverse problems. IEEE transactions on medical imaging, 38(2), 394-405.
            - Davis, S. P., Kumar, S., Alexandrov, Y., Bhargava, A., da Silva Xavier, G., Rutter, G. A., ... & McGinty, J. (2019). Convolutional neural networks for reconstruction of undersampled optical projection tomography data applied to in vivo imaging of zebrafish. Journal of biophotonics, 12(12), e201900128.
        """

        if self.denoiser_method == "U-Net":

            self.dw = unet.UNet(self.unet_options)

        elif self.denoiser_method == "resnet":

            if self.use_shared_weights == True:
                self.dw = dw(self.resnet_options)
            else:
                self.dw = nn.ModuleList([dw(self.resnet_options) for _ in range(self.K)])


def normalize_images(images):
    """Normalize images to zero mean and unit variance.

    Args:
        images (Tensor): Shape (B, 1, H, W).

    Returns:
        Tensor: Normalized images.
    """

    image_norm = torch.zeros_like(images)

    for i, image in enumerate(images):

        # print(image.max())
        image = (image - image.mean()) / image.std()
        # image_norm[i, ...] = (image - image.min()) / (image.max() - image.min())
        image_norm[i, ...] = image


    return image_norm


"""
Process sinograms in 2D
"""



for k, v in os.environ.items():
    if k.startswith("QT_") and "cv2" in v:
        del os.environ[k]


class Rec_Modes(Enum):
    """Supported reconstruction modes."""
    FBP_CPU = 0
    FBP_GPU = 1
    TWIST_CPU = 2
    TOMODL_CPU = 3
    TOMODL_GPU = 4
    UNET_CPU = 5
    UNET_GPU = 6


class Order_Modes(Enum):
    """Axis ordering of sinograms:
        Vertical   → (theta, Q, Z)
        Horizontal → (Q, theta, Z)
    """
    Vertical = 0
    Horizontal = 1


def my_filtering_function(pair):
    unwanted_key = "num_batches"
    key, value = pair
    if unwanted_key in key:
        return False  # filter pair out of the dictionary
    else:
        return True  # keep pair in the filtered dictionary


class OPTProcessor:
    """High-level OPT reconstruction controller.

    Handles:
        • Sinogram resizing
        • CPU/GPU FBP
        • TwIST reconstruction
        • ToMoDL reconstruction
        • U-Net-based reconstruction
        • Correct axis ordering
        • Rotation mode handling

    Attributes include:
        resize_val, rec_process, order_mode, clip_to_circle,
        use_filter, batch_size, iterations, invert_color, etc.
    """
    def __init__(self):
        """
        Variables for OPT processor
        """

        self.resize_val = 100
        self.rec_process = Rec_Modes.FBP_CPU.value
        self.order_mode = Order_Modes.Vertical.value
        self.clip_to_circle = False
        self.use_filter = True  
        self.batch_size = 1
        self.is_half_rotation = False
        self.filter_FBP = "ramp" # can be  "shepp-logan" or "cosine" or "hamming" or "hann" 
        self.is_resize = False
        self.iradon_functor = None
        self.invert_color = False
        self.iterations = 6
        self.set_reconstruction_process()

    def set_reconstruction_process(self):

        if self.is_half_rotation == True:
            rotation_factor = 1
        else:
            rotation_factor = 2

        # This should change depending on the method
        if self.rec_process == Rec_Modes.FBP_CPU.value:

            self.angles_gen = lambda num_angles: np.linspace(0, rotation_factor * 180, num_angles, endpoint=False)

        elif self.rec_process == Rec_Modes.FBP_GPU.value or self.rec_process == Rec_Modes.TOMODL_GPU.value:

            assert torch.cuda.is_available() == True
            self.angles_gen = lambda num_angles: np.linspace(0, rotation_factor * np.pi, num_angles, endpoint=False)


    def resize(self, sinogram_volume: np.ndarray, type_sino="3D"):
        """Resize sinogram before reconstruction.

        Args:
            sinogram_volume (ndarray): Input sinogram.
            type_sino (str): "3D" or "2D".

        Returns:
            ndarray: Resized sinogram.
        """

        if self.order_mode == Order_Modes.Vertical.value:
            self.theta, self.Q, self.Z = sinogram_volume.shape
        elif self.order_mode == Order_Modes.Horizontal.value:
            self.Q, self.theta, self.Z = sinogram_volume.shape

        if self.clip_to_circle == True:
            sinogram_size = self.resize_val
        else:
            sinogram_size = int(np.ceil(self.resize_val * np.sqrt(2)))
        print("orginal sinogram range: ", sinogram_volume.min(), sinogram_volume.max())
        if self.order_mode == Order_Modes.Vertical.value:

            # sinogram_resize = np.zeros((self.theta, sinogram_size, self.Z), dtype=np.float32)
            sinogram_resize = resize_skimage(sinogram_volume, (self.theta, sinogram_size, self.Z), preserve_range=True)

        elif self.order_mode == Order_Modes.Horizontal.value:

            # sinogram_resize = np.zeros((sinogram_size, self.theta, self.Z), dtype=np.float32)
            sinogram_resize = resize_skimage(sinogram_volume, (sinogram_size, self.theta, self.Z), preserve_range=True)
        print("resized sinogram range: ", sinogram_resize.min(), sinogram_resize.max())
        # for idx in tqdm.tqdm(range(self.Z)):

        #     if self.order_mode == Order_Modes.Vertical.value:
        #         sinogram_resize[:, :, idx] = cv2.resize(
        #             sinogram_volume[:, :, idx],
        #             (sinogram_size, self.theta),
        #             interpolation=cv2.INTER_NEAREST,
        #         )

        #     elif self.order_mode == Order_Modes.Horizontal.value:
        #         sinogram_resize[:, :, idx] = cv2.resize(
        #             sinogram_volume[:, :, idx],
        #             (self.theta, sinogram_size),
        #             interpolation=cv2.INTER_NEAREST,
        #         )



        return sinogram_resize

    def reconstruct(self, sinogram: np.ndarray):
        """Reconstruct a sinogram using the selected method.

        Supported modes include:
            - FBP_CPU
            - FBP_GPU
            - TWIST_CPU
            - TOMODL_CPU / TOMODL_GPU
            - UNET_CPU / UNET_GPU

        Args:
            sinogram (ndarray): Input sinogram.

        Returns:
            ndarray: Reconstructed image volume.
        """
        if self.is_half_rotation == True:
            rotation_factor = 1
        else:
            rotation_factor = 2
        # give the angles in radians
        self.angles_torch = np.linspace(0, rotation_factor * np.pi, self.theta, endpoint=False)
        self.angles = np.linspace(0, rotation_factor * 180, self.theta, endpoint=False)


        if self.iradon_functor == None:
            try:
                self.angles_torch = np.linspace(0, rotation_factor * np.pi, self.theta, endpoint=False)
                self.iradon_functor = radon_thrad(
                    thetas=self.angles_torch,
                    circle=self.clip_to_circle,
                    filter_name=None if self.use_filter == False else self.filter_FBP,
                    device=device,
                )
            except:
                self.iradon_functor = None
                self.angles_torch = None

        if self.rec_process == Rec_Modes.FBP_GPU.value:
            self.iradon_functor = radon_thrad(
                thetas=self.angles_torch,
                circle=self.clip_to_circle,
                filter_name=None if self.use_filter == False else self.filter_FBP,
                device=device,
            )

            # sino have shape (Q, theta, Z) => (Z, 1, Q, theta)
            # then we have the iradon to be (Z, 1, Q, Q) => (Q, Q, Z)

            def _iradon(sino):
                sino = sino.transpose(2, 0, 1)
                sino = torch.from_numpy(sino[:, None, :, :]).to(device)
                reconstruction = self.iradon_functor.filter_backprojection(sino)
                # reconstruction = normalize_images(reconstruction)
                reconstruction = reconstruction.permute(1, 2, 3, 0)[0].cpu()
                reconstruction = np.asarray(reconstruction.numpy())

                # save reconstruction to tif file
                # tif.imwrite("reconstruction_FBP_GPU.tif", reconstruction)
                # print("mean and std of reconstruction: ", reconstruction[0, :, :].mean(), reconstruction[0, :, :].std())
                return reconstruction

            self.iradon_function = _iradon

        elif self.rec_process == Rec_Modes.FBP_CPU.value:
            self.iradon_function = lambda sino: iradon_scikit(
                sino[..., 0],
                self.angles,
                circle=self.clip_to_circle,
                filter_name=None if self.use_filter == False else self.filter_FBP,
            )[..., None]

        elif self.rec_process == Rec_Modes.TOMODL_GPU.value:

            resnet_options_dict = {
                "number_layers": 8,
                "kernel_size": 3,
                "features": 64,
                "in_channels": 1,
                "out_channels": 1,
                "stride": 1,
                "use_batch_norm": True,
                "init_method": "xavier",
                "device": device,
            }

            self.tomodl_dictionary = {
                "use_torch_radon": True,
                "metric": "psnr",
                "K_iterations": self.iterations,
                "number_projections_total": sinogram.shape[0],
                "acceleration_factor": 10,
                "image_size": sinogram.shape[1],
                "lambda": 0.5,
                "is_half_rotation": self.is_half_rotation,
                "use_shared_weights": True,
                "denoiser_method": "resnet",
                "resnet_options": resnet_options_dict,
                "in_channels": 1,
                "out_channels": 1,
                "device": device,
                "iter_conjugate": 10,
            }

            self.iradon_functor = ToMoDL(self.tomodl_dictionary)

            __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
            # artifact_path = os.path.join(__location__, "modl_dark_unnormal.ckpt")
            # artifact_path = os.path.join(__location__, "tomodl256_3.ckpt")
            artifact_path = os.path.join(__location__, "weights_tomodl100.ckpt")
            tomodl_checkpoint = torch.load(artifact_path, map_location=torch.device("cuda:0"), weights_only=True)

            ########################### old weight loading ############################
            # tomodl_checkpoint["state_dict"] = {
            #     k.replace("model.", ""): v for (k, v) in tomodl_checkpoint["state_dict"].items()
            # }

            # self.iradon_functor.load_state_dict(
            #     dict(filter(my_filtering_function, tomodl_checkpoint["state_dict"].items()))
            # )
            ############################# fix lambda weight loading ############################
            tomodl_checkpoint = {
                k.replace("model.", ""): v for k, v in tomodl_checkpoint.items()
            }
            # tomodl_checkpoint["state_dict"] = dict(filter(my_filtering_function, tomodl_checkpoint["state_dict"].items()))
            self.iradon_functor.load_state_dict(tomodl_checkpoint, strict=False)
            self.iradon_functor.eval()
            # print(sinogram.shape)
            # print("lambda is: ", self.iradon_functor.lam)
            #######################################################################################

            radon24 = radon_thrad(self.angles_torch, circle=self.clip_to_circle, filter_name=None, device=device)


            # the self.iradon_functor receive a reconstructed image (B, 1, Q, Q)
            # the input is a sinogram (B, 1, Q, theta)
            def _iradon(sino):
                with torch.inference_mode():
                    sino = sino.transpose(2, 0, 1)
                    sino = torch.from_numpy(sino[:, None, :, :]).to(device)
                    sino = ramp_filter_torch(sino, device=device)
                    reconstruction = radon24.filter_backprojection(sino)
                    # save mean and std of reconstruction for each image in the batch
                    # make reconstruction to cpu to apply median filter
                    reconstruction_new = reconstruction.clone().cpu().numpy()
                    mean_reconstruction, std_reconstruction = mean_std_median_filter(reconstruction_new)

                    mean_reconstruction = torch.from_numpy(mean_reconstruction)
                    std_reconstruction = torch.from_numpy(std_reconstruction)
                    # median filter the reconstruction 
                    reconstruction = normalize_images(reconstruction)  
                    # median filter the reconstruction
                    output = self.iradon_functor(reconstruction)[
                        "dc" + str(self.tomodl_dictionary["K_iterations"])
                    ].cpu()
                    output = normalize_images(output)
                    # undo the normalization with mean and std of reconstruction
                    output = output * std_reconstruction.unsqueeze(1).unsqueeze(1) + mean_reconstruction.unsqueeze(1).unsqueeze(1)
                    # output shape (B, 1, Q, Q)
                    output = np.asarray(output.numpy())
                    # print("mean and std of reconstruction: ", output[0, 0, :, :].mean(), output[0, 0, :, :].std())
                    output = output.transpose(1, 2, 3, 0)[0]


                    return output

            self.iradon_function = _iradon

        elif self.rec_process == Rec_Modes.TOMODL_CPU.value:
            resnet_options_dict = {
                "number_layers": 8,
                "kernel_size": 3,
                "features": 64,
                "in_channels": 1,
                "out_channels": 1,
                "stride": 1,
                "use_batch_norm": True,
                "init_method": "xavier",
                "device": torch.device("cpu"),
            }

            self.tomodl_dictionary = {
                "use_torch_radon": True,
                "metric": "psnr",
                "K_iterations": self.iterations,
                "number_projections_total": sinogram.shape[0],
                "acceleration_factor": 10,
                "image_size": sinogram.shape[1],
                "lambda": 0.5,
                "is_half_rotation": self.is_half_rotation,
                "use_shared_weights": True,
                "denoiser_method": "resnet",
                "resnet_options": resnet_options_dict,
                "in_channels": 1,
                "out_channels": 1,
                "device": torch.device("cpu"),
                "iter_conjugate": 10,
            }

            self.iradon_functor = ToMoDL(self.tomodl_dictionary)
            __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
            # artifact_path = os.path.join(__location__, "tomodl256_3.ckpt")
            artifact_path = os.path.join(__location__, "weights_tomodl100.ckpt")
            tomodl_checkpoint = torch.load(artifact_path, map_location=torch.device("cpu"), weights_only=True)

            tomodl_checkpoint= {
                k.replace("model.", ""): v for (k, v) in tomodl_checkpoint.items()
            }

            self.iradon_functor.load_state_dict(
                dict(filter(my_filtering_function, tomodl_checkpoint.items()))
            )
            self.iradon_functor.eval()
            radon24 = radon_thrad(self.angles_torch, circle=self.clip_to_circle, filter_name=None, device="cpu")

            def _iradon(sino):
                with torch.inference_mode():
                    sino = sino.transpose(2, 0, 1)
                    sino = torch.from_numpy(sino[:, None, :, :])
                    sino = ramp_filter_torch(sino, device="cpu")
                    reconstruction = radon24.filter_backprojection(sino)
                    # save mean and std of reconstruction for each image in the batch
                    # make reconstruction to cpu to apply median filter
                    reconstruction_new = reconstruction.clone().numpy()
                    mean_reconstruction, std_reconstruction = mean_std_median_filter(reconstruction_new)

                    mean_reconstruction = torch.from_numpy(mean_reconstruction)
                    std_reconstruction = torch.from_numpy(std_reconstruction)
                    # median filter the reconstruction 
                    reconstruction = normalize_images(reconstruction)  
                    # median filter the reconstruction
                    output = self.iradon_functor(reconstruction)[
                        "dc" + str(self.tomodl_dictionary["K_iterations"])
                    ]
                    output = normalize_images(output)
                    # undo the normalization with mean and std of reconstruction
                    output = output * std_reconstruction.unsqueeze(1).unsqueeze(1) + mean_reconstruction.unsqueeze(1).unsqueeze(1)
                    # output shape (B, 1, Q, Q)
                    output = np.asarray(output.numpy())
                    # print("mean and std of reconstruction: ", output[0, 0, :, :].mean(), output[0, 0, :, :].std())
                    output = output.transpose(1, 2, 3, 0)[0]


                    return output

            self.iradon_function = _iradon

        elif self.rec_process == Rec_Modes.TWIST_CPU.value:

            Psi = lambda x, th: TVdenoise(x, 2 / th, 3)
            #  set the penalty function, to compute the objective
            Phi = lambda x: TVnorm(x)

            twist_dictionary = {
                "LAMBDA": 1e-4,
                "TOLERANCEA": 1e-4,
                "STOPCRITERION": 1,
                "VERBOSE": 1,
                "INITIALIZATION": 0,
                "MAXITERA": 10000,
                "GPU": 0,
                "PSI": Psi,
                "PHI": Phi,
            }

            A = lambda x: radon_scikit(x, self.angles, circle=self.clip_to_circle)
            AT = lambda sino: iradon_scikit(sino, self.angles, circle=self.clip_to_circle)
            self.iradon_function = lambda sino: TwIST(
                sino[..., 0], A, AT, 0.01, twist_dictionary, true_img=AT(sino[..., 0])
            )[0][..., None]

        elif self.rec_process == Rec_Modes.UNET_GPU.value:
            radon24 = radon_thrad(self.angles_torch, circle=self.clip_to_circle, filter_name=None, device=device)

            self.iradon_functor = UNet(
                n_channels=1, n_classes=1, residual=True, up_conv=True, batch_norm=True, batch_norm_inconv=True
            ).to(device)
            # TODO: load the weights of the unet

            def _iradon(sino):
                sino = sino.transpose(2, 0, 1)
                sino = torch.from_numpy(sino[:, None, :, :]).to(device)
                # sino = ramp_filter_torch(sino, device=device)
                reconstruction = radon24.filter_backprojection(sino)
                output = self.iradon_functor(reconstruction).detach().cpu()
                output = np.asarray(output.numpy())
                return output.transpose(1, 2, 3, 0)[0]

            self.iradon_function = _iradon

        elif self.rec_process == Rec_Modes.UNET_CPU.value:
            radon24 = radon_thrad(self.angles_torch, circle=self.clip_to_circle, filter_name=None, device="cpu")
            self.iradon_functor = UNet(
                n_channels=1, n_classes=1, residual=True, up_conv=True, batch_norm=True, batch_norm_inconv=True
            ).to("cpu")
            # TODO: load the weights of the unet

            def _iradon(sino):
                sino = sino.transpose(2, 0, 1)
                sino = torch.from_numpy(sino[:, None, :, :])
                # sino = ramp_filter_torch(sino, device="cpu")
                reconstruction = radon24.filter_backprojection(sino)
                output = self.iradon_functor(reconstruction).detach().cpu()
                output = np.asarray(output.numpy())
                return output.transpose(1, 2, 3, 0)[0]
            self.iradon_function = _iradon


        reconstruction = self.iradon_function(sinogram)
        
        if self.invert_color == True:
            reconstruction = reconstruction.max() - reconstruction

        return reconstruction

    # def correct_and_reconstruct(self, sinogram: np.ndarray, type_sino="3D"):
    #     """
    #     Corrects rotation axis by finding optimal registration via maximising reconstructed image's intensity variance.

    #     Based on 'Walls, J. R., Sled, J. G., Sharpe, J., & Henkelman, R. M. (2005). Correction of artefacts in optical projection tomography. Physics in Medicine & Biology, 50(19), 4645.'

    #     Params:
    #     - sinogram
    #     """

    #     while self.shift_step >= 1:
    #         shifts = np.arange(-self.max_shift, self.max_shift, self.shift_step) + self.center_shift
    #         image_std = []
    #         for i, shift in enumerate(shifts):
    #             if self.order_mode == Order_Modes.Vertical.value and type_sino == "3D":
    #                 shift_tuple = (0, shift, 0)
    #             elif self.order_mode == Order_Modes.Horizontal.value or type_sino == "2D":
    #                 shift_tuple = (shift, 0, 0)

    #             sino_shift = ndi.shift(sinogram, shift_tuple, mode="nearest")

    #             # Get image reconstruction
    #             shift_iradon = self.reconstruct(sino_shift)

    #             # Calculate variance
    #             image_std.append(np.std(shift_iradon))

    #         # Update shifts
    #         self.center_shift = shifts[np.argmax(image_std)]
    #         self.max_shift /= 5
    #         self.shift_step /= 2.5
    #         if self.order_mode == Order_Modes.Vertical.value and type_sino == "3D":
    #             sinogram = ndi.shift(sinogram, (0, self.center_shift, 0), mode="nearest")
    #         elif self.order_mode == Order_Modes.Horizontal.value or type_sino == "2D":
    #             sinogram = ndi.shift(sinogram, (self.center_shift, 0, 0), mode="nearest")

    #     # Restart shifts
    #     self.max_shift = 200
    #     self.shift_step = 10
    #     self.center_shift = 0
    #     return self.reconstruct(sinogram)

