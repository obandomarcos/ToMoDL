import numpy as np
import k3d

# impoer resize 3D image
import tifffile as tiff
import glob
import tqdm
from natsort import natsorted

from skimage.exposure import match_histograms
from skimage.transform import resize


def normalize_01(images):
    return (images - images.min()) / (images.max() - images.min())


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


ctf3 = natsorted(glob.glob("datasets/ctf3/*.tif"))
for image_path1 in tqdm.tqdm(ctf3[::10]):
    image1 = tiff.imread(image_path1)
    image2 = tiff.imread(image_path1.replace("ctf3", "ctf2"))
    image3 = tiff.imread(image_path1.replace("ctf3", "ctf1"))

    image2 = match_histograms(image2, image1, channel_axis=-1)
    image3 = match_histograms(image3, image1, channel_axis=-1)
    y_offset = 0
    img_height = image1.shape[0]
    stitched_image = np.zeros((img_height * 3, image1.shape[1]))
    x, y = 650, 700
    for i, img in enumerate([image1, image2, image3]):

        # stitched_image[y_offset:y_offset+img_height, :] = 0
        stitched_image[y_offset : y_offset + img_height, :] = img

        if i == 0:
            y_offset += img_height - x

        if i == 1:
            y_offset += img_height - y

    stitched_image = stitched_image[100 : -x - y - 450, :]
    stitched_image = stitched_image / flat_field_estimate(stitched_image)
    # downscale the image to 1/2
    stitched_image = resize(
        stitched_image, (stitched_image.shape[0] // 2, stitched_image.shape[1] // 2), anti_aliasing=True
    )
    stitched_image = normalize_01(stitched_image)
    # convert to 16 bit
    stitched_image = (stitched_image * 65535).astype(np.uint16, copy=False)
    tiff.imsave(image_path1.replace("ctf3", "stitched_2X_acc10x"), stitched_image)
