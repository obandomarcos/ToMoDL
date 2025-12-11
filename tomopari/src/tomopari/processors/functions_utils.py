"""
Tomographic preprocessing and fast reconstruction utilities.

This module provides helper functions for optical/CT tomographic workflows,
including:

• Flat-field estimation from corner/background regions  
• Sinogram resizing for memory-efficient reconstruction  
• Fast GPU-accelerated filtered back-projection using QBI-Radon  
• Automatic rotation-axis correction by maximizing reconstruction variance  
• Sinogram shifting and image normalization utilities

These functions support 2D and 3D sinograms in multiple axis orderings, and are
designed to integrate with PyTorch-based reconstruction pipelines.

Dependencies:
    • NumPy
    • SciPy (ndimage)
    • scikit-image (resize)
    • PyTorch
    • QBI-Radon (for GPU FBP reconstruction)
"""

import numpy as np
# import cv2
from skimage.transform import resize as resize_skimage
import tqdm
import scipy.ndimage as ndi
import torch
from QBI_radon import Radon


def flat_field_estimate(img, ratio_corners=0.03):
    """Estimate a flat-field (background) intensity from image corners.

    The function extracts several small corner and side regions from the image
    and computes their mean intensities, rejecting outliers before averaging.
    This is useful for correcting illumination inhomogeneities in tomography
    projections.

    Args:
        img (ndarray): Input 2D image.
        ratio_corners (float): Fraction of the image size used to define
            corner-region width. Default is 0.03 (3%).

    Returns:
        float: Estimated flat-field/background intensity.
    """
    height, width = img.shape
    # get corner size as 2% of the  dimension
    corner_size = int(min(height, width) * ratio_corners)

    # Extract the four corner regions (top-left, top-right, bottom-left, bottom-right)
    top_left = img[:corner_size, :corner_size]
    top_right = img[:corner_size, -corner_size:]
    bottom_left = img[-corner_size:, :corner_size]
    bottom_right = img[-corner_size:, -corner_size:]
    middle_left = img[height // 2 - corner_size // 2 : height // 2 + corner_size // 2, :corner_size]
    middle_right = img[height // 2 - corner_size // 2 : height // 2 + corner_size // 2, -corner_size:]
    corner_means = np.array(
        [
            top_left.mean(),
            top_right.mean(),
            bottom_left.mean(),
            bottom_right.mean(),
            middle_left.mean(),
            middle_right.mean(),
        ]
    )
    valid_corners = corner_means[
        (corner_means > np.percentile(corner_means, 10)) & (corner_means < np.percentile(corner_means, 90))
    ]
    flat_field_estimate = valid_corners.mean()

    return flat_field_estimate



def resize_sino(sinogram_volume: np.ndarray, order_mode=0, resize_val=100, clip_to_circle=False):
    """Resize sinograms before reconstruction.

    This function rescales the detector dimension of a 3D sinogram volume to a
    target size, optionally clipping to a circle-inscribed region to reduce
    memory footprint.

    Args:
        sinogram_volume (ndarray): Input sinogram volume shaped either as:
            order_mode = 0 → (theta, detector, z)
            order_mode = 1 → (detector, theta, z)
        order_mode (int): Dimension ordering of the sinogram volume.
        resize_val (int): Target detector size.
        clip_to_circle (bool): If True, resizes directly to `resize_val`;
            otherwise resizes to ceil(resize_val * sqrt(2)) to preserve corners.

    Returns:
        ndarray: Resized sinogram volume with preserved intensity range.
    """

    if order_mode == 0:
        theta, Q, Z = sinogram_volume.shape
    elif order_mode == 1:
        Q, theta, Z = sinogram_volume.shape

    if clip_to_circle == True:
        sinogram_size = resize_val
    else:
        sinogram_size = int(np.ceil(resize_val * np.sqrt(2)))

    if order_mode == 0:

        # sinogram_resize = np.zeros((theta, sinogram_size, Z), dtype=np.float32)
        sinogram_resize = resize_skimage(sinogram_volume, (theta, sinogram_size, Z), preserve_range=True)

    elif order_mode == 1:
        # sinogram_resize = np.zeros((sinogram_size, theta, Z), dtype=np.float32)
        sinogram_resize = resize_skimage(sinogram_volume, (sinogram_size, theta, Z), preserve_range=True)

    return sinogram_resize

def fast_reconstruct_FBP(sinogram: np.ndarray, resize_val=None, batch_process=32,device="cpu", rotation_factor=2, order_mode=0, clip_to_circle=False):
    """Perform fast filtered back-projection (FBP) reconstruction.

    Uses the QBI-Radon GPU/PyTorch implementation of the inverse Radon transform
    to reconstruct each slice in batches for memory efficiency.

    Args:
        sinogram (ndarray): Input 3D sinogram volume.
        resize_val (int or None): Optional detector-size resizing before FBP.
        batch_process (int): Number of slices reconstructed per batch.
        device (str): 'cpu' or GPU device string (e.g. 'cuda:0').
        rotation_factor (float): Scaling factor determining projection angle
            coverage. Example: rotation_factor = 2 → angles span 0–2π.
        order_mode (int): 0 → (theta, detector, z), 1 → (detector, theta, z).
        clip_to_circle (bool): If True, reconstruction assumes circular support.

    Returns:
        ndarray: Reconstructed 3D volume with shape (z, x, y).

    Notes:
        Internally performs:
            • Optional resizing
            • Dimension reordering if required
            • Batch GPU FBP using Radon.filter_backprojection()
    """
    if order_mode == 0:
        theta, Q, Z = sinogram.shape
    elif order_mode == 1:
        Q, theta, Z = sinogram.shape
    iradon_functor = Radon(thetas=np.linspace(0, rotation_factor * np.pi, theta, endpoint=False),
                        circle=clip_to_circle,
                        filter_name="ramp",
                        device=device)

    opt_volume = []
    if resize_val is not None:
        sinogram = resize_sino(sinogram, order_mode=order_mode, resize_val=resize_val, clip_to_circle=clip_to_circle)

    if order_mode == 0:
        sinogram = sinogram.transpose(1, 0, 2)

    slice_reconstruction = range(Z)
    batch_start = slice_reconstruction[0]
    batch_end = batch_start + batch_process
    while batch_start <= slice_reconstruction[-1]:
        # print("Reconstructing slices {} to {}".format(batch_start, batch_end), end="\r")
        zidx = slice(batch_start, batch_end)
        sino_batch = sinogram[:, :, zidx]
        sino_batch = sino_batch.transpose(2, 0, 1)

        sino_batch = torch.from_numpy(sino_batch[:, None, :, :]).to(device)
        reconstruction = iradon_functor.filter_backprojection(sino_batch)
        # reconstruction = normalize_images(reconstruction)
        reconstruction = reconstruction.permute(1, 2, 3, 0)[0].cpu()
        reconstruction = np.asarray(reconstruction.numpy())
        opt_volume.append(reconstruction)
        batch_start = batch_end
        batch_end += batch_process
    opt_volume = np.concatenate(opt_volume, axis=-1)
    opt_volume = np.rollaxis(opt_volume, -1)
    return opt_volume


def find_center_shift(sinogram: np.ndarray, bar_thread=None, rotation_factor=2, type_sino="3D", order_mode=0, clip_to_circle=False, device="cpu"):
    """Estimate and correct rotation-axis misalignment in a sinogram.

    The method scans candidate detector-axis shifts, reconstructs slices using
    FBP, and selects the shift that maximizes variance in the reconstruction,
    following the approach of:

        Walls et al., "Correction of artefacts in optical projection tomography,"
        Physics in Medicine & Biology (2005).

    Args:
        sinogram (ndarray): 2D or 3D sinogram.
        bar_thread: Optional progress-bar object with `.start()`, `.run()`.
        rotation_factor (float): Projection angle scaling for FBP.
        type_sino (str): "3D" or "2D" sinogram mode.
        order_mode (int): Sinogram dimension ordering.
        clip_to_circle (bool): Whether reconstruction assumes circular support.
        device (str): FBP computation device ('cpu' or 'cuda').

    Returns:
        ndarray: Sinogram corrected for rotation-axis shift.

    Notes:
        • The search reduces automatically in step size for coarse-to-fine
          refinement.
        • Variance of reconstructed images is used as the objective metric.
        • Supports GPU-based reconstruction through fast_reconstruct_FBP().
    """

    if order_mode == 0:
        theta, Q, Z = sinogram.shape
    elif order_mode == 1:
        Q, theta, Z = sinogram.shape

    if Q > 300:
        new_sinogram = resize_sino(sinogram, order_mode=order_mode, resize_val=100, clip_to_circle=clip_to_circle)
    else:
        new_sinogram = sinogram
        
    if order_mode == 0:
        theta, Q, Z = new_sinogram.shape
        factor_shift = sinogram.shape[1] / Q
    elif order_mode == 1:
        Q, theta, Z = new_sinogram.shape
        factor_shift = sinogram.shape[0] / Q

    # get the 70% of the sinogram from the center
    # new_sinogram = new_sinogram[:, :, int(Z * 0.2):int(Z * 0.7)]
    # print("new_sinogram shape: ", new_sinogram.shape)
    # max_shift is the number of pixels to shift, take 10 pecent of the sinogram size
    max_shift = min(int(Q * 0.1), 200)
    shift_step = 2
    center_shift = 0
    # take only xx percent slices from sinogram from the center

    print("max_shift: ", max_shift, "shift_step: ", shift_step, )
    # reduce the number of theta to 100 to reduce the memory usage
    if theta > 100:
        factor_theta = theta // 100
        if order_mode == 0:
            new_sinogram = new_sinogram[::factor_theta, :, :]
        elif order_mode == 1:
            new_sinogram = new_sinogram[:, ::factor_theta, :]


    # calculate batch_process base on theta x Q size for optimizing GPU resource 
    # should be the power of 2 nearest to the result of 1073741 / (theta * Q) with int 

    if bar_thread is not None:
        bar_thread.start()
        bar_thread.max = 2 *max_shift
        bar_thread.value = 0
        bar_thread.run()
    center_shift = 0
    while shift_step >= 1:
        shifts = np.arange(-max_shift, max_shift, shift_step) + center_shift
        image_std = []
        for shift in tqdm.tqdm(shifts):
            if order_mode == 0 and type_sino == "3D":
                shift_tuple = (0, shift, 0)
            elif order_mode == 1 or type_sino == "2D":
                shift_tuple = (shift, 0, 0)

            sino_shift = ndi.shift(new_sinogram, shift_tuple, mode="nearest")

            # Get image reconstruction
            shift_iradon = fast_reconstruct_FBP(sino_shift, device=device, rotation_factor=rotation_factor, order_mode=order_mode, clip_to_circle=False, 
                                                batch_process=32)
            if order_mode == 1:
                print("shift: ", shift, "std: ", np.std(shift_iradon))
            # Calculate varianceshift
            image_std.append(np.std(shift_iradon))
            # create a progress bar thread to update the progress bar from -max_shift to max_shift with shift_step
            if bar_thread is not None:
                bar_thread.value = shift + shift_step + max_shift - center_shift
                bar_thread.run()
        # Update shifts
        center_shift = shifts[np.argmax(image_std)]
        max_shift /= 4
        shift_step /= 2
        
        if bar_thread is not None:
            bar_thread.value = 0
            bar_thread.max = 2 * max_shift
            bar_thread.run()
        print("center_shift: ", center_shift, "max_shift: ", max_shift, "shift_step: ", shift_step)
    print("final shift: ", center_shift * factor_shift)
    if order_mode == 0 and type_sino == "3D":
        sinogram = ndi.shift(sinogram, (0, center_shift * factor_shift, 0), mode="nearest")
    elif order_mode == 1 or type_sino == "2D":
        sinogram = ndi.shift(sinogram, (center_shift * factor_shift, 0, 0), mode="nearest")
    if bar_thread is not None:
        bar_thread.value = 0
        bar_thread.run()
    return sinogram