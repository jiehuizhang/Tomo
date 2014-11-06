'''Mass & Architecture Distortion 3D '''

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
from pylab import *
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


'''Process in sequence'''
def Mass3dExtra(im,classifier):

    sliceList = []

    for i in range(im.size_2):

        imdata = im.data[i]

        # parameters
        params = [8, 20, 0.0185,0.9]       
        sampRate = 30
        winSize = 15

        # gabor kernel and filtering
        kernels = gabor_filter.creat_Gabor_Kernels(params[0],params[1],params[2],params[3])
        response = gabor_filter.compute_Response(imdata,kernels)

        # response analysis
        (batchResp, integratedResp) = ra.cerat_batch_response(response,sampRate,winSize)
        poll = ra.vote(batchResp)
        integrated_poll = ra.integrating_poll(poll,sampRate,winSize,response[0].shape)

        # feature computing
        imageSlice = TImage.TImageSlice()
        patches = fex.patch_Extraction(imdata,poll,i,sampRate,90,threshold=16.4)
        
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
            imageSlice.LightPatchList.append(lightPatch)

        # data projecting
        
        false_lab = np.zeros((slice_feats.shape[0],0))
        data_projected = Dimreduction.dim_Reduction(slice_feats, false_lab, opt ='randtree',
                                                        n_components=2, visualize = False)
        classifier.classify(data_projected)
        imageSlice.predicts = classifier.predicts
        imageSlice.feats = slice_feats

        sliceList.append(imageSlice)
    
    return 

'''Process in parallel'''
def parallel_Mass_Extra(i,imdata,classifier):
    outputPath = None
    if platform.system() == 'Windows':
        outputPath = 'C:/Tomosynthesis/localtest/res/'	 
    else:
        outputPath = '/home/yanbin/Tomosynthesis/results/5016/'
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
    tiffLib.imsave(outputPath + str(i) + 'dialated.tif',np.float32(dilated))	
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
    
    #tiffLib.imsave(outputPath + str(i) + 'dialated.tif',np.float32(dilated))

    # gabor kernel and filtering
    kernels = gabor_filter.creat_Gabor_Kernels(params[0],params[1],params[2],params[3])
    response = gabor_filter.compute_Response(imdata,kernels)

    # response analysis
    (batchResp, integratedResp) = ra.cerat_batch_response(response,sampRate,winSize)
    poll = ra.vote(batchResp)
    tiffLib.imsave(outputPath + str(i) + 'poll___.tif',np.float32(poll))

    # remove pectoral muscle
    poll = PMHoughT.PMremove(poll, visulization = False)
    tiffLib.imsave(outputPath + str(i) + 'poll.tif',np.float32(poll))
    
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

    # data projecting
    
    if slice_feats != None:
        '''
        if slice_feats.shape[0] < slice_feats.size:
            false_lab = np.zeros((slice_feats.shape[0],))
            data_projected = Dimreduction.dim_Reduction(slice_feats, false_lab, opt ='randtree',
                                                        n_components=2, visualize = False)
            	
            classifier.classify(data_projected)
            imageSlice.predicts = classifier.predict
        '''
    print ('done slice ', i, 'patches ', len(patches))
    return imageSlice


def parallelWrapper(args):

    return parallel_Mass_Extra(*args)


'''Process in parallel using manager and proxies '''
def parallel_MassExtra_proxy(im,i,classifier):
     
    # parameters
    params = [4, 20, 0.0185,0.9]       
    sampRate = 30
    winSize = 15

    # gabor kernel and filtering
    kernels = gabor_filter.creat_Gabor_Kernels(params[0],params[1],params[2],params[3])
    response = gabor_filter.compute_Response(im.data[i],kernels)

    # response analysis
    (batchResp, integratedResp) = ra.cerat_batch_response(response,sampRate,winSize)
    poll = ra.vote(batchResp)
    integrated_poll = ra.integrating_poll(poll,sampRate,winSize,response[0].shape)

    # feature computing
    patches = fex.patch_Extraction(im.data[i],poll,i,sampRate,90,threshold=7.5)
    for k in range(len(patches)):
        
        patches[k].getRingsFeats(numrings = 8)
        patches[k].getHOGeats(numsector=36)
        patches[k].getFDFeats()
        patches[k].feats = np.hstack((patches[k].rings_feats,
                                          patches[k].FD_feats,patches[k].hog_feats))
        if k == 0:
            slice_feats = patches[k].feats
        else:
            slice_feats = np.vstack((slice_feats,patches[k].feats))

    im.patchesList[i] = patches
    im.feats[i] = slice_feats
    
    # data projecting
    false_lab = np.zeros((slice_feats.shape[0],0))
    data_projected = Dimreduction.dim_Reduction(slice_feats, false_lab, opt ='randtree',
                                                    n_components=2, visualize = False)
    classifier.classify(data_projected)
    im.predicts[i] = classifier.predicts

    outputPath = 'C:/Tomosynthesis/localtest/res/'
    np.savetxt(outputPath + 'feats_' + str(i) + '.txt', np.asarray(im.feats[i]), delimiter='\t')

class ScriptManager(BaseManager):
    pass

def process(lock,im,i,classifier):
    with lock:
        parallel_MassExtra_proxy(im,i,classifier)
def parallel_MassExtra_manager(im,classifier): 

    ScriptManager.register('Image', TImage.TImage, exposed=['patchesList', 'feats','predicts'])
    manager = ScriptManager()
    manager.start()
    manager.Image = im
    
    lock = Lock()
    p_1 = Process(target=process, name='process_1', args=(lock,manager.Image,0,classifier))
    p_1.start()
    p_2 = Process(target=process, name='process_2', args=(lock,manager.Image,1,classifier))
    p_2.start()

    p_1.join()
    p_2.join()
    '''process_q = []
    num_cpu = 3
    i = 0
    while i < 9:
        for cpu in range(num_cpu):
            p_cpu = Process(target=parallel_MassExtra_proxy, name='process_'+str(cpu), args=(im,i,classifier))
            p_cpu.start()
            i = i+1
            process_q.append(p_cpu)

        for cpu_release in range(num_cpu):
            process_q[cpu_release].join()'''

    print im.size_2
    print im.feats[0]
            
    

    



    
