"""Micro cacification detection"""
import math
import numpy as np
from scipy import ndimage
import scipy.ndimage.filters as filters
import tiffLib

def bs_Estimatimation(imdata, threshold, depth = 16):

    height, width = imdata.shape
    scale = int(math.pow(2,depth))
    npixel = width*height
    
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
    cdf = np.zeros(scale, dtype=np.float32)
    cdf[0] = prob[0]
    for i in range(1,scale-1):
        cdf[i] = prob[i] + cdf[i-1]
        if cdf[i]> threshold:
            estimated_bs = i
            break

    return estimated_bs

def fg_thresholding(imdata, threshold):

    fg = np.zeros(imdata.shape, dtype=np.uint16)
    fg[:,:] = imdata[:,:]
    index = fg < threshold
    fg[index] = threshold
    return fg

    
def log_filtering(imdata, winSize,sigma, fg_thresh,option = 'proplog'):
    '''sample_rate should be less than window size'''
    
    sample_rate = winSize
    nrow, ncol = imdata.shape
    rows = np.array(range(winSize,nrow,sample_rate))
    cols = np.array(range(winSize,ncol,sample_rate))

    rsu = np.maximum(rows - winSize,np.zeros(len(rows), dtype=np.int))
    rsd = np.minimum(rows + winSize,np.ones(len(rows), dtype=np.int)*(nrow-1))
    csl = np.maximum(cols - winSize,np.zeros(len(cols), dtype=np.int))
    csr = np.minimum(cols + winSize,np.ones(len(cols), dtype=np.int)*(ncol-1))

    log_response = np.zeros(imdata.shape, dtype=np.double)
    for rs in range(len(rows)):
        for cs in range(len(cols)):
            # extract data
            block = imdata[rsu[rs]:rsd[rs],csl[cs]:csr[cs]]

            # extract foreground
            estimated_bs = bs_Estimatimation(block, fg_thresh)
            fg = fg_thresholding(block,estimated_bs)

            # compute log response 
            temp_response = np.zeros(block.shape, np.double)
            filters.gaussian_laplace(fg, sigma, output=temp_response, mode='reflect')

            # composite response
            
            ub = rsu[rs] + winSize/2
            loc_ub = winSize/2
            db = rsd[rs] - winSize/2 
            loc_db = 3*winSize/2 
            lb = csl[cs] + winSize/2 
            loc_lb = winSize/2 
            rb = csr[cs] - winSize/2 
            loc_rb = 3*winSize/2
            # upper bound
            if rsu[rs] == 0:
                ub = 0
                loc_ub = 0
            # lower bound
            if rsd[rs] == nrow-1:
                db = nrow-1
                loc_db = temp_response.shape[0]
            # left bound
            if csl[cs] == 0:
                lb = 0
                loc_lb = 0
            # right bound
            if csr[cs] == ncol - 1:
                rb = ncol - 1
                loc_rb = temp_response.shape[1]
            
            log_response[ub:db,lb:rb] = temp_response[loc_ub:loc_db, loc_lb:loc_rb]

    if option == 'log':
        return -log_response
    if option == 'proplog':
        log_rep_prop = 1000*log_response / imdata
        return -log_rep_prop

def laebl_connecte_comp(imdata, threshold, size_constrain):

    # calculated connected labels
    mask = imdata > threshold
    label_im, nb_labels = ndimage.label(mask)

    # remove objects out of size_constrain
    sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))
    mask_size1 = sizes < size_constrain[0]
    mask_size2 = sizes > size_constrain[1]
    mask_size = mask_size1 + mask_size2
    remove_pixel = mask_size[label_im]
    label_im[remove_pixel] = 0

    # clear out regions in the original image
    rm_imdata = np.zeros(imdata.shape, dtype=np.double)
    rm_imdata[:,:] = imdata[:,:]
    index = label_im == 0
    rm_imdata[index] = 0

    return rm_imdata

    
    





            
