"""The denoising module includes three steps:

*  Anscombe Transform
*  Adaptive Wiener Filter
*  Inverse Anscombe Transform (Unbiased)
"""

from scipy import signal
import numpy as np
import tiffLib
import math

def DenoisingAW(imdata,opt = 'asymptotic', block_m=5,block_n=5):
    """ The denoising main function.

    Parameters
    ----------
    imdata: numpy array
        The input image array
    opt: str
        The options for inverse transform. Default set as 'asymptotic'
    block_m: integer
        The window size_x for winnier filter
    block_n: integer
        The window size_y for winnier filter

    Examples
    --------
    >>> import ImageIO
    >>> import AT_denoising
    >>> dataPath = 'C:/Tomosynthesis/localtest/'
    >>> fileName = 'test-crop.tif'
    >>> im = ImageIO.imReader(dataPath,fileName, 'tif',2)
    >>> denoised = AT_denoising.DenoisingAW(im.data[0])
    
    """
    imtransformed = AnscombeTrans(imdata)
    imfiltered = AdaptWiener(imtransformed,block_m,block_n)
    iminversed = InAnscombeTrans(imfiltered,opt)

    return iminversed

def AnscombeTrans(imdata):
    """ The Anscombe Transform function.

    Parameters
    ----------
    imdata: numpy array
        The input image array
    """
    imdata = np.float32(imdata)
    z = 2*np.sqrt(imdata+3/8)

    return np.uint16(z)
    
def InAnscombeTrans(imdata, opt = 'exact'):
    """ The Inverse Anscombe Transform function.

    Parameters
    ----------
    imdata: numpy array
        The input image array
    opt: str
        The options for inverse transform. Default set as 'asymptotic'.
    """

    imdata = np.float32(imdata)
    if opt == 'algebra':
        z = imdata*imdata/4 - 3/8
        
    if opt == 'asymptotic':
        z = imdata*imdata/4 - 1/8

    if opt == 'exact':
        z = imdata*imdata/4 + math.sqrt(3/2)/imdata/4 - 11/8/(imdata*imdata) + \
            5/8*math.sqrt(3/2)/(imdata*imdata*imdata) - 1/8
        z = np.maximum(z, np.zeros(imdata.shape, dtype=np.float32))

    if opt == 'MMSE':
        print 'sth'
        
    return np.uint16(z)   

def AdaptWiener(imdata,block_m=5,block_n=5):
    """ The Inverse Anscombe Transform function.

    Parameters
    ----------
    imdata: numpy array
        The input image array
    block_m: integer
        The window size_x for winnier filter
    block_n: integer
        The window size_y for winnier filter
    """

    shape = imdata.shape
    npixel_nhood = block_m*block_n
    structure = np.ones((block_m,block_n), dtype=np.uint64)

    # Avoid overflow, convert original to double depth
    imdata = np.uint64(imdata)
    
    # Estimate the local mean of f
    localMean = signal.convolve2d(imdata, structure, mode='same', boundary='fill', fillvalue=0)
    localMean = np.uint64(localMean/npixel_nhood)

    # Estimate the local Variance of f
    localVar = signal.convolve2d(imdata*imdata, structure, mode='same', boundary='fill', fillvalue=0)
    localVar = localVar/npixel_nhood - localMean*localMean

    # Estimate the noise power
    noise = np.mean(localVar)

    # Convert to float to handle negative values
    imdata = np.float32(imdata)
    localMean = np.float32(localMean)
    localVar = np.float32(localVar)
    noise = np.float32(noise)

    # The formula
    f = imdata - localMean
    g = localVar - noise

    g = np.maximum(g, np.zeros(shape, dtype=np.float32))
    localVar = np.maximum(localVar, noise*np.ones(shape, dtype=np.float32))
    f = f / localVar
    f = f * g
    f = f + localMean

    return np.uint16(f)
