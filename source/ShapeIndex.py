""" From Surface shape and curvature scales """
from PIL import ImageFilter
import numpy as np
import scipy.ndimage
import math
import tiffLib

def ShapeIndex(imdata, gaussianRadious,medianRadious = 1):
    ''' The formula is:
        
                                        dnx_x + dny_y
        s = 2 / PI * arctan ---------------------------------------
                             sqrt((dnx_x - dny_y)^2 + 4 dny_x dnx_y)
        
        where _x and _y are the x and y components of the
        partial derivatives of the normal vector of the surface
        defined by the intensities of the image.
        
        n_x and n_y are the negative partial derivatives of the
        intensity, approximated by simple differences.'''

    # Gaussian Blur
    smoothdata = scipy.ndimage.filters.gaussian_filter(imdata, gaussianRadious)
    tiffLib.imsave('/home/yanbin/Tomosynthesis/code/' + 'blured.tif',smoothdata)

    # Index Mapping using formula
    dx = deriveX(smoothdata)
    dy = deriveY(smoothdata)
    dxx = deriveX(dx)
    dxy = deriveY(dx)
    dyx = deriveX(dy)
    dyy = deriveY(dy)
    
    factor = 2 / np.pi

    shape = imdata.shape
    width = shape[1]
    height = shape[0]
    indexmap = np.zeros(shape, dtype=np.float32)
    for j in range(width):
        for i in range(height):
            dnx_x = -dxx[i][j]
            dnx_y = -dxy[i][j]
            dny_x = -dyx[i][j]
            dny_y = -dyy[i][j]
            denom = math.sqrt((dnx_x - dny_y) * (dnx_x - dny_y) + 4 * dnx_y * dny_x)
            indexmap[i][j] = factor * math.atan((dnx_x + dny_y) / denom)
                
    # Remove NAN using Median Filter
    for j in range(width):
        for i in range(height):
            var = indexmap[i][j]
            if math.isnan(var):
                ledge = max(0,j-medianRadious)
                redge = min(width, j+medianRadious)
                uedge = max(0,i-medianRadious)
                dedge = min(height, i+medianRadious)
                count = 0
                summ = 0.0
                for jj in range(ledge,redge):
                    for ii in range(uedge,dedge):
                        temvar = indexmap[ii][jj]
                        if math.isnan(temvar)== False:
                            summ = summ + temvar
                        count = count + 1
                summ = summ / count
                indexmap[i][j] = summ
                        
    return indexmap      

def deriveX(imdata):
    '''Calculate the derivative along X axis'''
    shape = imdata.shape
    dx = np.zeros(shape, dtype=np.float32)

    width = shape[1]
    height = shape[0]
    for i in range(height):
        previous = 0.0
        for j in range(width):
            current = imdata[i][j]          
            diff = np.float32(current) - previous
            dx[i][j] = diff
            previous = current

    return dx  

def deriveY(imdata):
    '''Calculate the derivative along Y axis'''
    shape = imdata.shape
    dy = np.zeros(shape, dtype=np.float32)

    width = shape[1]
    height = shape[0]
    for j in range(width):
        previous = 0.0
        for i in range(height):
            current = imdata[i][j]
            diff = np.float32(current) - previous
            dy[i][j] = diff
            previous = current

    return dy
