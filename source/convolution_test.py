import ImageIO
import TImage
import tiffLib

from scipy import ndimage as nd
import scipy
from skimage.filter import gabor_kernel
import numpy as np
import time
from numpy.fft import fft, ifft, fft2, ifft2, fftshift

dataPath = 'C:/Tomosynthesis/localtest/'
fileName = 'test-crop.tif'
outputPath = 'C:/Tomosynthesis/localtest/res/'
im = ImageIO.imReader(dataPath,fileName, 'tif',2)


kernel = np.real(gabor_kernel(0.0185, 0, 20, 20/float(0.9)))
print kernel.shape

start = time.clock()
temp_response = nd.convolve(im.data[0], kernel, mode='nearest')
elapsed = (time.clock() - start)
print elapsed

start = time.clock()
data = np.lib.pad(im.data[0], ((0,kernel.shape[0]),(0,kernel.shape[1])),'edge')
temp_response_2 = np.fft.irfft2(np.fft.rfft2(data) * np.fft.rfft2(kernel,data.shape))
temp_response_2 = temp_response_2[kernel.shape[0]/2:data.shape[0] - kernel.shape[0]/2,kernel.shape[1]/2:data.shape[1] - kernel.shape[1]/2]
elapsed = (time.clock() - start)
print elapsed
print data.shape

tiffLib.imsave(outputPath + 'convolution.tif',np.float32(temp_response))
tiffLib.imsave(outputPath + 'fft.tif',np.float32(temp_response_2))
#tiffLib.imsave(outputPath + 'fftcolvele.tif',np.float32(temp_response_3))
tiffLib.imsave(outputPath + 'kernel.tif',np.float32(kernel))
tiffLib.imsave(outputPath + 'padded.tif',np.float32(data))
#tiffLib.imsave(outputPath + 'fftim.tif',np.float32(np.fft.fft2(data)))
#tiffLib.imsave(outputPath + 'fftkernel.tif',np.float32(np.fft.fft2(kernel, data.shape)))


