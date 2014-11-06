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
    seg_feats = np.zeros((len(file_list),7),np.double)
    all_feats = np.zeros((len(file_list),15),np.double)

    LightPatchList = []
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

        # Calculating segment features
        if opt == 'seg' or opt == 'all':
            
            patch = TPatch.TPatch()
            patch.initialize(im.data[0])
            seg_feats[counter,:] = patch.getSegmentFeats()

        if opt == 'all':
            all_feats[counter,:] = np.hstack((int_feats[counter,:], gr_feats[counter,:], seg_feats[counter,:]))

            lightPatch = TPatch.TLightPatch()
            lightPatch.pdata = im.data[0]
            lightPatch.feats = all_feats[counter,:]
            lightPatch.patch_center = (im.data[0].shape[0]/2, im.data[0].shape[1]/2)
            LightPatchList.append(lightPatch)

        counter = counter + 1
        
    if opt == 'all':
        return LightPatchList

    if opt == 'Int':
        return int_feats
    if opt == 'Grad':         
        return gr_feats

    if opt == 'seg':
        return seg_feats

def creatTrainigSam_3D(dataPath, iRnum = 6,iSnum = 12, gRnum = 4,gSnum = 12):

    file_list = os.listdir(dataPath)

    LightPatchList = []
    bagid = 0
    instanceid = 0
    for fil in file_list:
        im = ImageIO.imReader(dataPath, fil,'tif',3)

        for i in range(im.size_2):

            # Calculating intensity features
            patch = TPatch.TPatch()
            patch.initialize(im.data[i])
            int_feats = patch.getIntenfeats(iRnum,iSnum)

            # Calculating segment features         
            seg_feats = patch.getSegmentFeats()

            # Calculating gradient features
            im.downSample(rate = 2)
            eqimg = histEqualization.histEqualization(im.sampled_data[i], 16)
            smoothimg = filters.gaussian_filter(eqimg, sigma = 2, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0)
            patch = TPatch.TPatch()
            patch.initialize(smoothimg)
            gr_feats = patch.getGradfeats(gRnum,gSnum)
            
            feats = np.hstack((int_feats, gr_feats, seg_feats))

            lightPatch = TPatch.TLightPatch()
            lightPatch.pdata = im.data[i]
            lightPatch.feats = feats
            lightPatch.patch_center = (im.data[i].shape[0]/2, im.data[i].shape[1]/2)
            lightPatch.bagID = bagid
            lightPatch.instanceID = instanceid
            LightPatchList.append(lightPatch)

            instanceid = instanceid + 1
        
        bagid = bagid + 1

    return LightPatchList


























            
