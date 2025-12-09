import numpy as np
# import cv2
from skimage.transform import resize as resize_skimage
import tqdm
import scipy.ndimage as ndi
import torch
from QBI_radon import Radon


def flat_field_estimate(img, ratio_corners=0.03):

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
    """
    Resizes sinogram prior to reconstruction.
    Args:
        -sinogram_volume (np.ndarray): array to resize in any mode specified.
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
    # for idx in range(Z):

    #     if order_mode == 0:
    #         sinogram_resize[:, :, idx] = cv2.resize(
    #             sinogram_volume[:, :, idx],
    #             (sinogram_size, theta),
    #             interpolation=cv2.INTER_NEAREST,
    #         )

    #     elif order_mode == 1:
    #         sinogram_resize[:, :, idx] = cv2.resize(
    #             sinogram_volume[:, :, idx],
    #             (theta, sinogram_size),
    #             interpolation=cv2.INTER_NEAREST,
    #         )

    return sinogram_resize

def fast_reconstruct_FBP(sinogram: np.ndarray, resize_val=None, batch_process=32,device="cpu", rotation_factor=2, order_mode=0, clip_to_circle=False):
    """
    Fast reconstruction function.
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
    """
    Corrects rotation axis by finding optimal registration via maximising reconstructed image's intensity variance.

    Based on 'Walls, J. R., Sled, J. G., Sharpe, J., & Henkelman, R. M. (2005). Correction of artefacts in optical projection tomography. Physics in Medicine & Biology, 50(19), 4645.'

    Params:
    - sinogram
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