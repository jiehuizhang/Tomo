""" Creat Training Samples from image crop"""

import sys, os, re, operator, pdb, subprocess, time
import numpy as np
from scipy import ndimage
import scipy.ndimage.filters as filters
import ImageIO
import TPatch

import histEqualization
import AT_denoising

def creatTrainigSam(dataPath,opt = 'Rings', numrings = 10):

    # Calculating Ring features
    if opt == 'Rings':      
        
        file_list = os.listdir(dataPath)
        Rings_feats = np.zeros((len(file_list),2),np.double)
        counter = 0
        for fil in file_list:
            im = ImageIO.imReader(dataPath, fil,'tif',2)

            patch = TPatch.TPatch()
            patch.initialize(im.data[0])
            patch.getRings(numrings)
            patch.getRingsMeanFeats()
            patch.getRingsVarFeats()
            patch.LinearRegRingMeanFeats()
            patch.LinearRegRingVarFeats()
            Rings_feats[counter][0] = patch.ringLinearSlopeMean
            Rings_feats[counter][1] = patch.ringLinearSlopeVar

            counter = counter + 1

        return Rings_feats

    
    # Calculating Fractal features
    if opt == 'FD':
        
        file_list = os.listdir(dataPath)
        Fd_feats = np.zeros((len(file_list),4),np.double)
        counter = 0
        for fil in file_list:
            im = ImageIO.imReader(dataPath, fil,'tif',2)
            
            eqimg = histEqualization.histEqualization(im.data[0], 16)
            denoised = ndimage.filters.gaussian_filter(eqimg, sigma = 3, order=0, output=None, mode='nearest')
            #denoised = AT_denoising.DenoisingAW(eqimg,opt = 'asymptotic', block_m=10,block_n=10)

            patch = TPatch.TPatch()
            patch.initialize(denoised)
            fds = patch.getFD()
            Fd_feats[counter][0] = fds[0]
            Fd_feats[counter][1] = fds[1]
            Fd_feats[counter][2] = fds[2]
            Fd_feats[counter][3] = fds[3]
            counter = counter + 1
        return Fd_feats

    # Calculating HOG features
    if opt == 'HOG':
        
        file_list = os.listdir(dataPath)
        hog_feats = np.zeros((len(file_list),3),np.double)
        for i in range(len(file_list)):
        
            im = ImageIO.imReader(dataPath, file_list[i],'tif',2)

            # downsampling
            im.downSample(rate = 2)
    
            # histogram equalization
            eqimg = histEqualization.histEqualization(im.sampled_data[0], 16)
    
            # smoothing
            im.sampled_data[0].shape
            smoothimg = filters.gaussian_filter(eqimg, sigma = 2, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0)
            
            patch = TPatch.TPatch()
            patch.initialize(smoothimg)
            patch.computeGradient()
            patch.gradOrieNormalize(threshold = 1500)
            patch.getGSectors(36)
            patch.getNormLevl()
            patch.getNormGradmagnitude()
            hog_feats[i][0] = patch.norGra_level_mean
            hog_feats[i][1] = patch.norGra_level_max
            hog_feats[i][2] = patch.norGra_level_std

        return hog_feats



























            
