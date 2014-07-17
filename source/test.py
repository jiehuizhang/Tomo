import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from scipy.misc import lena

from skimage.feature import hog
from skimage import data, color, exposure
import numpy as np

import tiffLib
import ImageIO
import AT_denoising

dataPath = 'C:/Tomosynthesis/localtest/'
outputPath = 'C:/Tomosynthesis/localtest/res/'
fileName = 'smoothed_local.tif'
im = ImageIO.imReader(dataPath,fileName, 'tif',2)
    
image = color.rgb2gray(data.lena())
image = im.data[0]

denoised = AT_denoising.DenoisingAW(im.data[0],opt = 'asymptotic', block_m=3,block_n=3)
#image = denoised
fd, hog_image = hog(image, orientations=8, pixels_per_cell=(6, 6),
                    cells_per_block=(1, 1), visualise=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

ax1.axis('off')
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Input image')

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))


ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
plt.show()

def sector_mask(shape,centre,radius,angle_range):
    """
    Return a boolean mask for a circular sector. The start/stop angles in  
    `angle_range` should be given in clockwise order.
    """

    x,y = np.ogrid[:shape[0],:shape[1]]
    cx,cy = centre
    tmin,tmax = np.deg2rad(angle_range)

    # ensure stop angle > start angle
    if tmax < tmin:
            tmax += 2*np.pi

    # convert cartesian --> polar coordinates
    r2 = (x-cx)*(x-cx) + (y-cy)*(y-cy)
    theta = np.arctan2(x-cx,y-cy) - tmin

    # wrap angles between 0 and 2*pi
    theta %= (2*np.pi)

    # circular mask
    circmask = r2 <= radius*radius

    # angular mask
    anglemask = theta <= (tmax-tmin)

    return circmask*anglemask

matrix = lena()
mask = sector_mask(matrix.shape,(200,100),300,(0,50))
matrix[~mask] = 0
plt.imshow(matrix)
plt.show()
