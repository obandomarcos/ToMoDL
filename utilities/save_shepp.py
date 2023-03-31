
import numpy as np
from skimage import data, img_as_float
from skimage.transform import radon, iradon
import matplotlib.pyplot as plt
import cv2

# Generate Shepp-Logan phantom
phantom = data.shepp_logan_phantom()
phantom = cv2.imread('/home/obanmarcos/Balseiro/DeepOPT/napari-tomodl/sinossss.jpg', cv2.IMREAD_GRAYSCALE)
# Define sinogram parameters
theta = np.linspace(0., 2*180., phantom.shape[0], endpoint=False)
sinogram_shape = (len(theta), phantom.shape[1])

sinogram = iradon(phantom.T, theta=theta, circle=True)
                      
cv2.imwrite(f'volume/irad.jpg', 255*sinogram)