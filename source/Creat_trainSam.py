""" Creat Training Samples from image crop"""

import sys, os, re, operator, pdb, subprocess, time
import numpy as np
from scipy import ndimage
import scipy.ndimage.filters as filters
import ImageIO
import TPatch

import histEqualization
import AT_denoising
import tiffLib

def creatTrainigSam(dataPath, opt = 'all', iRnum = 6,iSnum = 12, gRnum = 4,gSnum = 12):

    file_list = os.listdir(dataPath)
    int_feats = np.zeros((len(file_list),4),np.double)
    gr_feats = np.zeros((len(file_list),4),np.double)
    all_feats = np.zeros((len(file_list),8),np.double)

    counter = 0
    for fil in file_list:
        im = ImageIO.imReader(dataPath, fil,'tif',2)

        # Calculating intensity features
        if opt == 'Int' or opt == 'all':
            patch = TPatch.TPatch()
            patch.initialize(im.data[0])
            int_feats[counter,:] = patch.getIntenfeats(iRnum,iSnum)

        # Calculating gradient features
        if opt == 'Grad' or opt == 'all':
            # preprocess
            im.downSample(rate = 2)
            eqimg = histEqualization.histEqualization(im.sampled_data[0], 16)
            smoothimg = filters.gaussian_filter(eqimg, sigma = 2, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0)

            patch = TPatch.TPatch()
            patch.initialize(smoothimg)
            gr_feats[counter,:] = patch.getGradfeats(gRnum,gSnum)

        

        if opt == 'all':
            all_feats[counter,:] = np.hstack((int_feats[counter,:], gr_feats[counter,:] ))

        counter = counter + 1

    if opt == 'all':
        return all_feats

    if opt == 'Int':
        return int_feats
    if opt == 'Grad':         
        return gr_feats

























            
