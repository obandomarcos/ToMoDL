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
from typing import Union
import cv2
import dask.array as da
from skimage.transform import resize as resize_skimage

torch._dynamo.config.suppress_errors = True
torch._dynamo.config.capture_scalar_outputs = True

from skimage.transform import radon as radon_scikit
from skimage.transform import iradon as iradon_scikit
from skimage.filters import median
from skimage.transform import resize as resize_skimage
from skimage.morphology import disk
from .unet import AttU_Net

from .alternating import TwIST, TVdenoise, TVnorm
import numpy as np
import os

# import cv2
from enum import Enum

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


# Modify for multi-gpu
device = torch.device("cuda" if use_torch_radon == True else "cpu")


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
        # image_median = image[0]
        mean_images[i, 0] = image_median.mean()
        std_images[i, 0] = image_median.std()
    return mean_images, std_images


class dwLayer(nn.Module):
    """
    Creates denoiser singular layer
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
            # self.batch_norm = nn.BatchNorm2d(self.weights_size[1])
            self.norm = nn.GroupNorm(num_groups=1, num_channels=self.weights_size[1])

    def forward(self, x):
        """
        Forward pass for block
        Params:
            - x (torch.Tensor): Image batch to be processed
        """
        output = self.conv(x)

        if self.use_batch_norm:

            output = self.norm(output)

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

            self.nw["c" + str(i)] = dwLayer(self.dw_layer_dict).to(device)

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
    """
    This class is created to do the data-consistency (DC) step as described in paper.
    """

    def __init__(self, kw_dictionary):
        """
        Initializes Conjugate gradients step.
        Params:
            - kw_dictionary (dict): Keyword dictionary
        """

        self.number_projections = kw_dictionary["number_projections"]
        self.lam = kw_dictionary["lambda"]
        self.use_torch_radon = kw_dictionary["use_torch_radon"]
        self.device = kw_dictionary["device"]
        self.use_scikit = kw_dictionary["use_scikit"]
        self.angles = np.linspace(0, 2 * np.pi, self.number_projections, endpoint=True)
        self.iter_conjugate = kw_dictionary["iter_conjugate"]
        if self.use_torch_radon == True:
            self.radon = radon_thrad(thetas=self.angles, circle=True, device=self.device, filter_name=None)

    def forward(self, img):
        """
        Applies the operator (A^H A + lam*I) to image, where A is the forward Radon transform.
        Params:
            - img (torch.Tensor): Input tensor
        """

        sinogram = self.radon(img) / img.shape[-1]
        iradon = self.radon.filter_backprojection(sinogram) * np.pi / self.number_projections
        output = iradon + self.lam * img

        # print('output forward: {} {}'.format(output.max(), output.min()))
        # print('Term z max {}, min {}'.format((iradon/self.lam).max(), (iradon/self.lam).min()))
        # print('Term input max {}, min {}'.format(img.max(), img.min()))
        # print('Term output max {}, min {}'.format(output.max(), output.min()))
        return output

    def inverse(self, rhs):
        """
        Applies CG on the batch
        Params:
            - rhs (torch.Tensor): Right-hand side tensor for applying inversion of (A^H A + lam*I) operator
        """
        y = self.conjugate_gradients(self.forward, rhs)  # This indexing may fail

        return y

    def conjugate_gradients(self, A, rhs):
        """
        My implementation of conjugate gradients in PyTorch
        """

        i = 0
        x = torch.zeros_like(rhs)
        r = rhs
        p = rhs
        rTr = torch.sum(r * r)

        while (i < self.iter_conjugate) and torch.ge(rTr, 1e-4):

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
            number_projections (int)
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

        #####################################################################################
        for i in range(1, self.K + 1):
            j = str(i)
            self.out["dw" + j] = self.dw.forward(self.out["dc" + str(i - 1)])
            rhs = x / self.lam + self.out["dw" + j]

            self.out["dc" + j] = self.AtA.inverse(rhs)

            # self.out["dc" + j] = self.normalize_image_01(self.out["dc" + j])
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
        self.number_projections = kw_dictionary["number_projections"]
        self.acceleration_factor = kw_dictionary["acceleration_factor"]
        self.lam = kw_dictionary["lambda"]
        self.lam = torch.nn.Parameter(torch.tensor([self.lam], requires_grad=False, device=self.device))

        self.use_shared_weights = kw_dictionary["use_shared_weights"]
        self.denoiser_method = kw_dictionary["denoiser_method"]

        self.in_channels = kw_dictionary["in_channels"]
        self.out_channels = kw_dictionary["out_channels"]
        self.iter_conjugate = kw_dictionary["iter_conjugate"]

        if self.denoiser_method == "resnet":
            self.resnet_options = kw_dictionary["resnet_options"]
        self.AtA_dictionary = {
            "number_projections": self.number_projections,
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

            self.dw = AttU_Net()

        elif self.denoiser_method == "resnet":

            if self.use_shared_weights == True:
                self.dw = dw(self.resnet_options)
            else:
                self.dw = nn.ModuleList([dw(self.resnet_options) for _ in range(self.K)])


def normalize_images_zscore(images):
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


# for k, v in os.environ.items():
#     if k.startswith("QT_") and "cv2" in v:
#         del os.environ[k]


class Rec_Modes(Enum):
    """Supported reconstruction modes."""

    FBP_GPU = 0
    FBP_CPU = 1
    TOMODL_GPU = 2
    TOMODL_CPU = 3
    UNET_GPU = 4
    UNET_CPU = 5
    TWIST_CPU = 6


class Order_Modes(Enum):
    """Axis ordering of sinograms:
    Vertical   → (theta, Q, Z)
    Horizontal → (Q, theta, Z)
    """

    Vertical = 0
    Horizontal = 1


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

        self.resize_val = 128
        self.rec_process = Rec_Modes.FBP_CPU.value
        self.order_mode = Order_Modes.Vertical.value
        self.clip_to_circle = True
        self.use_filter = True
        self.batch_size = 1
        self.is_half_rotation = False
        self.ratio_circle = 1
        self.filter_FBP = "ramp"  # can be  "shepp-logan" or "cosine" or "hamming" or "hann"
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

    def resize_batch(
        self,
        sinogram_batch: Union[np.ndarray, da.Array],
    ) -> np.ndarray:
        """
        Resize one sinogram batch along the detector axis only.

        Expected input shapes
        ---------------------
        Vertical order:
            (theta, Q, batch_size)

        Horizontal order:
            (Q, theta, batch_size)

        Returns
        -------
        np.ndarray
            Resized, C-contiguous batch with dtype float32.

            Vertical order:
                (theta, resized_Q, batch_size)

            Horizontal order:
                (resized_Q, theta, batch_size)
        """

        # Compute only the selected Dask batch.
        if isinstance(sinogram_batch, da.Array):
            sinogram_batch = sinogram_batch.compute()

        if self.clip_to_circle:
            target_detector_size = int(self.resize_val)
        else:
            target_detector_size = int(np.ceil(self.resize_val * np.sqrt(2.0)))

        # ---------------------------------------------------------------
        # Put the detector axis first:
        #
        #     (theta, Q, batch) -> (Q, theta, batch)
        #
        # Horizontal mode is already in this format.
        # ---------------------------------------------------------------
        if self.order_mode == Order_Modes.Vertical.value:
            detector_first = np.moveaxis(
                sinogram_batch,
                1,
                0,
            )
            return_vertical = True

        elif self.order_mode == Order_Modes.Horizontal.value:
            detector_first = sinogram_batch
            return_vertical = False

        else:
            raise ValueError(f"Unsupported order mode: {self.order_mode}")

        detector_size, theta, batch_size = detector_first.shape

        # Nothing needs to be resized.
        if detector_size == target_detector_size:
            return np.ascontiguousarray(
                sinogram_batch,
                dtype=np.float32,
            )

        # ---------------------------------------------------------------
        # Convert:
        #
        #     (Q, theta, batch) -> (Q, theta * batch)
        #
        # The first dimension is the only dimension being resized.
        # ---------------------------------------------------------------
        resize_input = np.ascontiguousarray(
            detector_first.reshape(
                detector_size,
                theta * batch_size,
            )
        )

        # INTER_AREA is generally better and faster for downsampling.
        # INTER_LINEAR is suitable for upsampling.
        if target_detector_size < detector_size:
            interpolation = cv2.INTER_AREA
        else:
            interpolation = cv2.INTER_LINEAR

        resized_2d = cv2.resize(
            resize_input,
            dsize=(
                theta * batch_size,
                target_detector_size,
            ),
            interpolation=interpolation,
        )

        # OpenCV may collapse dimensions in unusual single-column cases.
        resized_2d = resized_2d.reshape(
            target_detector_size,
            theta * batch_size,
        )

        resized = resized_2d.reshape(
            target_detector_size,
            theta,
            batch_size,
        )

        # Restore the original axis order.
        if return_vertical:
            resized = np.moveaxis(
                resized,
                0,
                1,
            )

        return np.ascontiguousarray(
            resized,
            dtype=np.float32,
        )

    # def resize_batch(
    #     self,
    #     sinogram_batch: Union[np.ndarray, da.Array],
    #     type_sino: str = "3D",
    # ) -> np.ndarray:
    #     """
    #     Resize one sinogram batch along the detector axis only.

    #     Expected input shapes
    #     ---------------------
    #     Vertical order:
    #         (theta, Q, batch_size)

    #     Horizontal order:
    #         (Q, theta, batch_size)

    #     Returns
    #     -------
    #     np.ndarray
    #         Resized batch with dtype float32.

    #         Vertical order:
    #             (theta, resized_Q, batch_size)

    #         Horizontal order:
    #             (resized_Q, theta, batch_size)
    #     """

    #     if sinogram_batch.ndim != 3:
    #         raise ValueError(f"Expected a 3D sinogram batch, but received " f"shape {sinogram_batch.shape}.")

    #     # Compute only the selected batch, not the complete Dask volume.
    #     if isinstance(sinogram_batch, da.Array):
    #         sinogram_batch = sinogram_batch.compute()

    #     # Avoid float64 output and reduce memory usage.
    #     sinogram_batch = np.asarray(
    #         sinogram_batch,
    #         dtype=np.float32,
    #         order="C",
    #     )

    #     if self.clip_to_circle:
    #         sinogram_size = int(self.resize_val)
    #     else:
    #         sinogram_size = int(np.ceil(self.resize_val * np.sqrt(2)))

    #     if self.order_mode == Order_Modes.Vertical.value:
    #         theta, detector_size, batch_size = sinogram_batch.shape

    #         output_shape = (
    #             theta,
    #             sinogram_size,
    #             batch_size,
    #         )

    #     elif self.order_mode == Order_Modes.Horizontal.value:
    #         detector_size, theta, batch_size = sinogram_batch.shape

    #         output_shape = (
    #             sinogram_size,
    #             theta,
    #             batch_size,
    #         )

    #     else:
    #         raise ValueError(f"Unsupported order mode: {self.order_mode}")

    #     sinogram_resize = resize_skimage(
    #         sinogram_batch,
    #         output_shape=output_shape,
    #         order=1,
    #         mode="edge",
    #         preserve_range=True,
    #         anti_aliasing=sinogram_size < detector_size,
    #     )

    #     return sinogram_resize.astype(
    #         np.float32,
    #         copy=False,
    #     )

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
                    ratio_circle=self.ratio_circle,
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
                ratio_circle=self.ratio_circle,
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
                "number_projections": sinogram.shape[1],
                "acceleration_factor": 20,
                "lambda": 0.5,
                "is_half_rotation": self.is_half_rotation,
                "use_shared_weights": True,
                "denoiser_method": "resnet",
                "resnet_options": resnet_options_dict,
                "in_channels": 1,
                "out_channels": 1,
                "device": device,
                "iter_conjugate": 5,
            }

            self.iradon_functor = ToMoDL(self.tomodl_dictionary)

            __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
            if sinogram.shape[0] <= 256:
                artifact_path = os.path.join(__location__, "tomodl100_state_dict.ckpt")
            else:
                # artifact_path = os.path.join(__location__, "tomodl100_state_dict.ckpt")
                artifact_path = os.path.join(__location__, "tomodl500_sd.ckpt")

            tomodl_checkpoint = torch.load(artifact_path, map_location=device, weights_only=True)
            # tomodl_checkpoint["state_dict"] = {
            #     k.replace("model.", ""): v for k, v in tomodl_checkpoint["state_dict"].items()
            # }
            self.iradon_functor.load_state_dict(tomodl_checkpoint, strict=True)
            self.iradon_functor.to(device)
            self.iradon_functor.eval()
            # print(sinogram.shape)
            if sinogram.shape[0] > 256:
                original_lam = self.iradon_functor.lam.item()
                self.iradon_functor.lam = torch.nn.Parameter(
                    torch.tensor([original_lam * (sinogram.shape[0] / 512)], requires_grad=False, device=device)
                )
                print("lambda is: ", self.iradon_functor.lam)

            radon24 = radon_thrad(
                self.angles_torch, circle=self.clip_to_circle, ratio_circle=1, filter_name=None, device=device
            )

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
                    reconstruction = normalize_images_zscore(reconstruction)
                    # median filter the reconstruction
                    output = self.iradon_functor(reconstruction)[
                        "dc" + str(self.tomodl_dictionary["K_iterations"])
                    ].cpu()
                    output = normalize_images_zscore(output)
                    # # undo the normalization with mean and std of reconstruction
                    output = output * std_reconstruction.unsqueeze(1).unsqueeze(1) + mean_reconstruction.unsqueeze(
                        1
                    ).unsqueeze(1)
                    # output shape (B, 1, Q, Q)
                    output = np.asarray(output.numpy())
                    # if self.clip_to_circle:
                    #     det_count = sino.shape[2]
                    #     # create grid
                    #     grid_y, grid_x = np.meshgrid(
                    #         np.linspace(-1, 1, det_count), np.linspace(-1, 1, det_count), indexing="ij"
                    #     )
                    #     # create circle mask
                    #     reconstruction_circle = (grid_x**2 + grid_y**2) <= self.ratio_circle**2
                    #     # expand to batch dimension (like repeat in torch)
                    #     reconstructed_circle = np.repeat(
                    #         reconstruction_circle[None, None, :, :], output.shape[0], axis=0  # shape (1,1,H,W)
                    #     )
                    #     real_output = np.zeros((output.shape[0], output.shape[1], det_count, det_count))
                    #     start_idx = (det_count - output.shape[2]) // 2
                    #     end_idx = start_idx + output.shape[2]
                    #     real_output[:, :, start_idx:end_idx, start_idx:end_idx] = output
                    #     real_output[reconstructed_circle == 0] = 0
                    #     output = real_output
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
                "number_projections": sinogram.shape[1],
                "acceleration_factor": 20,
                "lambda": 0.5,
                "is_half_rotation": self.is_half_rotation,
                "use_shared_weights": True,
                "denoiser_method": "resnet",
                "resnet_options": resnet_options_dict,
                "in_channels": 1,
                "out_channels": 1,
                "device": torch.device("cpu"),
                "iter_conjugate": 5,
            }

            self.iradon_functor = ToMoDL(self.tomodl_dictionary)

            __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
            if sinogram.shape[0] < 256:
                artifact_path = os.path.join(__location__, "tomodl100_state_dict.ckpt")
            else:

                artifact_path = os.path.join(__location__, "tomodl500_sd.ckpt")
            tomodl_checkpoint = torch.load(artifact_path, map_location=torch.device("cpu"), weights_only=True)

            self.iradon_functor.load_state_dict(tomodl_checkpoint, strict=True)
            self.iradon_functor.to(torch.device("cpu"))
            self.iradon_functor.eval()

            radon24 = radon_thrad(
                self.angles_torch,
                circle=self.clip_to_circle,
                ratio_circle=1,
                filter_name=None,
                device="cpu",
            )

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
                    reconstruction = normalize_images_zscore(reconstruction)
                    # median filter the reconstruction
                    output = self.iradon_functor(reconstruction)[
                        "dc" + str(self.tomodl_dictionary["K_iterations"])
                    ].cpu()
                    output = normalize_images_zscore(output)
                    # # undo the normalization with mean and std of reconstruction
                    output = output * std_reconstruction.unsqueeze(1).unsqueeze(1) + mean_reconstruction.unsqueeze(
                        1
                    ).unsqueeze(1)
                    # output shape (B, 1, Q, Q)
                    output = np.asarray(output.numpy())
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
                "number_projections": sinogram.shape[1],
                "acceleration_factor": 20,
                "lambda": 0.5,
                "is_half_rotation": self.is_half_rotation,
                "use_shared_weights": True,
                "denoiser_method": "U-Net",
                "resnet_options": resnet_options_dict,
                "in_channels": 1,
                "out_channels": 1,
                "device": device,
                "iter_conjugate": 5,
            }
            self.iradon_functor = ToMoDL(self.tomodl_dictionary)

            __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
            if sinogram.shape[0] <= 128:
                artifact_path = os.path.join(__location__, "tomodl128_Unet_sd.ckpt")
            elif 128 < sinogram.shape[0] <= 256:
                artifact_path = os.path.join(__location__, "tomodl256_Unet_sd.ckpt")
            else:
                artifact_path = os.path.join(__location__, "tomodl512_Unet_sd.ckpt")
            tomodl_checkpoint = torch.load(artifact_path, map_location=device, weights_only=True)

            # tomodl_checkpoint["state_dict"] = {
            #     k.replace("model.", ""): v for k, v in tomodl_checkpoint["state_dict"].items()
            # }
            self.iradon_functor.load_state_dict(tomodl_checkpoint, strict=True)
            self.iradon_functor.to(device)
            self.iradon_functor.eval()

            radon24 = radon_thrad(
                self.angles_torch, circle=self.clip_to_circle, ratio_circle=1, filter_name=None, device=device
            )

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
                    reconstruction = normalize_images_zscore(reconstruction)
                    # median filter the reconstruction
                    output = self.iradon_functor(reconstruction)[
                        "dc" + str(self.tomodl_dictionary["K_iterations"])
                    ].cpu()
                    output = normalize_images_zscore(output)
                    # # undo the normalization with mean and std of reconstruction
                    output = output * std_reconstruction.unsqueeze(1).unsqueeze(1) + mean_reconstruction.unsqueeze(
                        1
                    ).unsqueeze(1)
                    # output shape (B, 1, Q, Q)
                    output = np.asarray(output.numpy())
                    output = output.transpose(1, 2, 3, 0)[0]

                    return output

            self.iradon_function = _iradon

        elif self.rec_process == Rec_Modes.UNET_CPU.value:
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
                "number_projections": sinogram.shape[1],
                "acceleration_factor": 20,
                "lambda": 0.5,
                "is_half_rotation": self.is_half_rotation,
                "use_shared_weights": True,
                "denoiser_method": "U-Net",
                "resnet_options": resnet_options_dict,
                "in_channels": 1,
                "out_channels": 1,
                "device": torch.device("cpu"),
                "iter_conjugate": 5,
            }

            self.iradon_functor = ToMoDL(self.tomodl_dictionary)

            __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
            if sinogram.shape[0] <= 128:
                artifact_path = os.path.join(__location__, "tomodl128_Unet_sd.ckpt")
            elif 128 < sinogram.shape[0] <= 256:
                artifact_path = os.path.join(__location__, "tomodl256_Unet_sd.ckpt")
            else:
                artifact_path = os.path.join(__location__, "tomodl512_Unet_sd.ckpt")
            tomodl_checkpoint = torch.load(artifact_path, map_location=torch.device("cpu"), weights_only=True)

            # tomodl_checkpoint["state_dict"] = {
            #     k.replace("model.", ""): v for k, v in tomodl_checkpoint["state_dict"].items()
            # }
            self.iradon_functor.load_state_dict(tomodl_checkpoint, strict=True)
            self.iradon_functor.to(torch.device("cpu"))
            self.iradon_functor.eval()

            radon24 = radon_thrad(
                self.angles_torch,
                circle=self.clip_to_circle,
                ratio_circle=1,
                filter_name=None,
                device="cpu",
            )

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
                    reconstruction = normalize_images_zscore(reconstruction)
                    # median filter the reconstruction
                    output = self.iradon_functor(reconstruction)[
                        "dc" + str(self.tomodl_dictionary["K_iterations"])
                    ].cpu()
                    output = normalize_images_zscore(output)
                    # # undo the normalization with mean and std of reconstruction
                    output = output * std_reconstruction.unsqueeze(1).unsqueeze(1) + mean_reconstruction.unsqueeze(
                        1
                    ).unsqueeze(1)
                    # output shape (B, 1, Q, Q)
                    output = np.asarray(output.numpy())
                    output = output.transpose(1, 2, 3, 0)[0]

                    return output

            self.iradon_function = _iradon

        reconstruction = self.iradon_function(sinogram)

        # if self.invert_color == True:
        #     reconstruction = reconstruction.max() - reconstruction

        return reconstruction
