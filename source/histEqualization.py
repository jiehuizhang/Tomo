'''This module perform the histogram equalization algorithm to enhance the consrast of images'''
import numpy as np
import math

def histEqualization(imdata, depth = 16):

    shape = imdata.shape
    width = shape[1]
    height = shape[0]
    scale = int(math.pow(2,depth))
    npixel = width*height
    alpha = float(scale)/float(npixel)

    # Generate histogram
    hist = np.zeros(scale, dtype=np.uint16)
    for j in range(width):
        for i in range(height):
            val = imdata[i][j]
            hist[val] = hist[val] + 1

    # Calculate probability
    prob = np.zeros(scale, dtype=np.float32)
    for i in range(scale):
        prob[i] = float(hist[i])/float(npixel)

    # Generate CDF
    cdf = np.zeros(scale, dtype=np.int32)
    cdf[0] = hist[0]
    for i in range(1,scale-1): 
        cdf[i] = hist[i] + cdf[i-1]

    # Scale histogram
    sc = np.zeros(scale, dtype=np.uint16)
    for i in range(scale):
        sc[i] = round(cdf[i]*alpha)

    # Generate equalized histogram
    eqhist = np.zeros(scale, dtype=np.uint16)
    for i in range(scale):
        eqhist[sc[i]] = eqhist[sc[i]] + hist[i]


    # Mapping
    eqim = np.zeros(shape, dtype=np.uint16)
    for j in range(width):
        for i in range(height):
            eqim[i][j] = sc[imdata[i][j]]

    return eqim
