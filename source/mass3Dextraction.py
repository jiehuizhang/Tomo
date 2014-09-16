'''Mass & Architecture Distortion 3D '''
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


'''Process in sequence'''
def Mass3dExtra(im,classifier):

    for i in range(im.size_2):
        print 'slice_' + str(i)
        
        image = im.data[i]
        
        # parameters
        params = [4, 20, 0.0185,0.9]       
        sampRate = 30
        winSize = 15

        # gabor kernel and filtering
        print 'filtering....'
        kernels = gabor_filter.creat_Gabor_Kernels(params[0],params[1],params[2],params[3])
        response = gabor_filter.compute_Response(image,kernels)

        # response analysis
        print 'responsing...'
        (batchResp, integratedResp) = ra.cerat_batch_response(response,sampRate,winSize)
        poll = ra.vote(batchResp)
        integrated_poll = ra.integrating_poll(poll,sampRate,winSize,response[0].shape)

        # feature computing
        print 'computing feats...'
        patches = fex.patch_Extraction(image,poll,i,sampRate,90,threshold=7.5)
        slice_feats = None
        for k in range(len(patches)):
            print k
            patches[k].getRingsFeats(numrings = 8)
            patches[k].getHOGeats(numsector=36)
            patches[k].getFDFeats()
            patches[k].feats = np.hstack((patches[k].rings_feats,
                                          patches[k].FD_feats,patches[k].hog_feats))
            print patches[k].hog_feats
            if k == 0:
                slice_feats = patches[k].feats
            else:
                slice_feats = np.vstack((slice_feats,patches[k].feats))
                
        im.feats[i] = slice_feats
        im.patchesList[i] = patches

        # data projecting
        false_lab = np.zeros((slice_feats.shape[0],0))
        data_projected = Dimreduction.dim_Reduction(slice_feats, false_lab, opt ='randtree',
                                                    n_components=2, visualize = False)
        # classifier.classify(data_projected)
        # im.predicts[i] = classifier.predicts

'''Process in parallel'''
def parallel_Mass_Extra(i,imdata,classifier):
     
    # parameters
    params = [4, 20, 0.0185,0.9]       
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
    patches = fex.patch_Extraction(imdata,poll,i,sampRate,90,threshold=7.5)
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

    # for debugging
    # outputPath = '/home/yanbin/localtest/'
    # np.savetxt(outputPath + str(i) + '_feats.txt', slice_feats, delimiter='\t')
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
            
    

    



    
