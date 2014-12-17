"""Pipeline for mass & architecture distortion extraction in 3D."""

import platform
import time
from multiprocessing import current_process, cpu_count
from multiprocessing import Manager, Process, Condition, Lock, Pool
from multiprocessing.managers import BaseManager
from Queue import Empty
import Dimreduction
import classification

import numpy as np
import scipy.ndimage.filters as filters
#from pylab import *
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from skimage.morphology import disk
from scipy import ndimage

import ImageIO
import TImage
import TPatch
import ShapeIndex
import histEqualization
import AT_denoising
from scipy import misc
import tiffLib
import gabor_filter
import Response_Analysis as ra
import bsckground_Substraction as bs
import feat_Extraction as fex
import PMHoughT

def parallel_Mass_Extra(i,imdata):
    """Extraction pipeline processed in parallel.

    Parameters
    ----------
    i: integer
        The current slice in the parallel processing pool
    im : TImage
        The entire image stack.
    """
    # parameters
    params = [8, 20, 0.0185,0.9]       
    sampRate = 30
    winSize = 15

    # skin-line remove & preproseesing
    threshold = 7200
    mask = imdata > threshold
    sline = np.zeros(imdata.shape, np.float32)
    sline[mask] = 1
    selem = disk(15)
    dilated = ndimage.convolve(sline, selem, mode='constant', cval=0)  
    mask = dilated > 0

    threshold2 = 2500
    mask2 = imdata < threshold2
    selem = disk(30)
    boundary = np.zeros(imdata.shape, np.float32)
    boundary[mask2] = 1
    dilated = ndimage.convolve(boundary, selem, mode='constant', cval=0)
    mask2 = dilated > 0
    
    imdata[mask] = 0
    imdata[mask2] = 0

    # gabor kernel and filtering
    kernels = gabor_filter.creat_Gabor_Kernels(params[0],params[1],params[2],params[3])
    response = gabor_filter.compute_Response(imdata,kernels)

    # response analysis
    (batchResp, integratedResp) = ra.cerat_batch_response(response,sampRate,winSize)
    poll = ra.vote(batchResp)
    #tiffLib.imsave(outputPath + str(i) + 'poll___.tif',np.float32(poll))

    # remove pectoral muscle
    poll = PMHoughT.PMremove(poll, visulization = False)
    #tiffLib.imsave(outputPath + str(i) + 'poll.tif',np.float32(poll))
    
    integrated_poll = ra.integrating_poll(poll,sampRate,winSize,response[0].shape)    	
    #tiffLib.imsave(outputPath + str(i) + 'integrated_poll.tif',np.float32(integrated_poll))

    # feature computing
    imageSlice = TImage.TImageSlice()
    patches = fex.patch_Extraction(imdata,poll,i,sampRate,90,threshold=16.4)   # 11.5
    
    slice_feats = None
    for k in range(len(patches)):
        
        # intensity ring features
        int_feats = patches[k].getIntenfeats(int_Rnum = 6,int_Snum = 12)

        # gradient sector features
        patches[k].downSampling(rate = 2)
        eqimg = histEqualization.histEqualization(patches[k].downsampled, 16)
        smoothimg = filters.gaussian_filter(eqimg, sigma = 2, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0)
        patch = TPatch.TPatch()
        patch.initialize(smoothimg)
        gr_feats = patch.getGradfeats(gr_Rnum = 4,gr_Snum = 12)

        # segment features            
        seg_feats = patches[k].getSegmentFeats()
        feats = np.hstack((int_feats,gr_feats,seg_feats))

        if k == 0:
            slice_feats = feats
        else:
            slice_feats = np.vstack((slice_feats,feats))

        lightPatch = TPatch.TLightPatch()
        lightPatch.image_center = patches[k].image_center
        lightPatch.pdata = patches[k].pdata
        lightPatch.patch_center = patches[k].patch_center
        lightPatch.feats = feats
        imageSlice.LightPatchList.append(lightPatch)
        imageSlice.feats = slice_feats

    print ('done slice ', i, 'patches ', len(patches))
    return imageSlice


def parallelWrapper(args):

    return parallel_Mass_Extra(*args)





    

    



    
