""" Gabor Filter Creation and convolution """

from skimage import data
from skimage.util import img_as_float
from skimage.filter import gabor_kernel

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as nd

def creat_Gabor_Kernels(norientation, sigma, frequency,gamma):
    '''norientation:    number of orientations, for calculating theta
       sigmm:           adjusting shap of kernel function (variance)
       frequency:       adjusting the wavelength'''

    kernels = []
    # looping of different orientation
    for orientation in range(norientation):
        theta = orientation / float(norientation) * np.pi
            
        kernel = np.real(gabor_kernel(frequency, theta=theta,
                                      sigma_x=sigma, sigma_y=sigma/float(gamma)))
        kernels.append(kernel)

    return kernels

def creat_FilterBank(norientation, sigmas, frequencies,gammas):

    filter_bank = []

    # looping of different sigmas
    for sigma in sigmas:

        # looping of different frequencies
        for frequency in frequencies:

            # looping of different gammas
            for gamma in gammas:
                filter_bank.append(creat_Gabor_Kernels(norientation, sigma, frequency,gamma))

    return filter_bank
        

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

def compute_Response(image,kernels):

    response = []
    for kernel in kernels:
        temp_response = nd.convolve(image, kernel, mode='constant', cval=0.0)
        response.append(temp_response)
    
    return response

def compute_Responses(image,filter_bank):

    responses = []
    for kernels in filter_bank:
        response = []
        for kernel in kernels:
            temp_response = nd.convolve(image, kernel, mode='constant', cval=0.0)
            response.append(temp_response)
        responses.append(response)
    
    return responses

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

'''#sigmas = []
#sigmas.append(1)
#filter_bank = creat_FilterBank(8,(1,2,3,4,5),(0.15,0.2,0.25),sigmas)
#plot_FilterBank(filter_bank)

shrink = (slice(0, None, 3), slice(0, None, 3))
brick = img_as_float(data.load('brick.png'))[shrink]

kernels = creat_Gabor_Kernels(8, 3, 0.1,1)
# plot_Kernels(kernels)

response = compute_Response(brick,kernels)
plot_Response(response)'''



                
        

    
