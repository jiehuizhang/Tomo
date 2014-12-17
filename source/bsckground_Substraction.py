
import scipy.ndimage
import numpy as np

def background_Substraction(imdata,sigma = 30):
    """ Background Substraction.

    Parameters
    ----------
    imdata: numpy array
        The input image array
    sigma: float
        The sigma value for gaussian filter.

    Examples
    --------
    >>> import ImageIO
    >>> import bsckground_Substraction as bs
    >>> dataPath = 'C:/Tomosynthesis/localtest/'
    >>> fileName = 'test-crop.tif'
    >>> im = ImageIO.imReader(dataPath,fileName, 'tif',2)
    >>> substracteed = bs.background_Substraction(im.data[0])

    """

    # Gaussian Blur
    smoothdata = scipy.ndimage.filters.gaussian_filter(imdata, sigma)

    # Substraction
    substracteed = np.abs(imdata - smoothdata)

    return substracteed
    
