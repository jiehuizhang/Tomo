""" This File includes functions that creat Gabor kernels
and compute the coresponding filtering response."""

from skimage.util import img_as_float
from skimage.filter import gabor_kernel
from scipy import ndimage as nd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

def creat_Gabor_Kernels(norientation, sigma, frequency,gamma):
    """
    This function creats the Gabor kernels with given parameters.
	
    norientation:    number of orientations
    sigmm:           scale of the kernel
    frequency:       wavelength/frequency of the kernel
    """

    kernels = []	
    for orientation in range(norientation):
        theta = orientation / float(norientation) * np.pi
            
        kernel = np.real(gabor_kernel(frequency, theta=theta,
                                      sigma_x=sigma, sigma_y=sigma/float(gamma)))
        kernels.append(kernel)

    return kernels

def creat_FilterBank(norientation, sigmas, frequencies, gammas):
    """
    This function creats the Gabor filter bank with given parameters.
	
    norientation:    number of orientations
    sigmas:           a list of scales of the kernels
    frequencies:        a list of the wavelength/frequency of the kernels
    """
    
    filter_bank = []

    for sigma in sigmas:

        for frequency in frequencies:

            for gamma in gammas:
                filter_bank.append(creat_Gabor_Kernels(norientation, sigma, frequency,gamma))

    return filter_bank

def fftconvolve(image, kernel):
    """
    This function calculate 2d convolution using fft
    image:   input image
    kernel:  input kernel
    """
    # padding so linear convolution is computed instead circular convolution
    data = np.lib.pad(image, ((0,kernel.shape[0]),(0,kernel.shape[1])),'edge')

    # fft computation
    response = np.fft.irfft2(np.fft.rfft2(data) * np.fft.rfft2(kernel,data.shape))

    # unpadding
    kr,kc = kernel.shape
    dr,dc = data.shape
    response = response[kr/2:dr - kr/2, kc/2:dc - kc/2]

    return response  
 
def compute_Response(image,kernels):
    """
    This function compute the filtering response of given
    image and kernels

    image:     Input image
    kernels:   Input kernels(a list of kernels with same
               scale but different orientations)
    """

    response = []
    for kernel in kernels:
        temp_response = fftconvolve(image, kernel)
        response.append(temp_response)
    
    return response 
	
def compute_Responses(image,filter_bank):
    """
    This function compute the filtering responses of given
    image and the filter bank

    image:     Input image
    filter_bank:   Input filter bank(a list of kernel set with different parameters)
    """

    responses = []
    for kernels in filter_bank:
        response = []
        for kernel in kernels:
            temp_response = fftconvolve(image, kernel)
            response.append(temp_response)
        responses.append(response)
    
    return responses

def plot_Kernels(kernels):

    nrows = len(kernels)/2
    ncols = 2

    fig, axes = plt.subplots(nrows, ncols)
    plt.gray()

    ker_index = 0
    for ax_row in axes:
        for ax_col in ax_row:
            ax_col.imshow(np.real(kernels[ker_index]), interpolation='nearest')
            ker_index = ker_index + 1
    plt.show()
	
def plot_FilterBank(filter_bank):

    nrows = len(filter_bank)
    ncols = len(filter_bank[0])
    
    fig, axes = plt.subplots(nrows, ncols)
    plt.gray()

    row_id = 0
    for ax_row in axes:
        col_id = 0
        for ax_col in ax_row:
            ax_col.imshow(np.real(filter_bank[row_id][col_id]), interpolation='nearest')
            col_id = col_id + 1
        row_id = row_id + 1
    plt.show()

def plot_Response(response):

    nrows = len(response)/2
    ncols = 2

    fig, axes = plt.subplots(nrows, ncols)
    plt.gray()

    resp_index = 0
    for ax_row in axes:
        for ax_col in ax_row:
            ax_col.imshow(response[resp_index], interpolation='nearest')
            resp_index = resp_index + 1
    plt.show()


