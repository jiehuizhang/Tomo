import time

import numpy as np

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

def main():

    ############################## single smv convert#######################
    '''dataPath = '/home/TomosynthesisData/Cancer_25_cases/5039/'
    outputPath = '/home/yanbin/Tomosynthesis/processed/5039/'
    fileName = '5039.recon08.smv'
    im = ImageIO.imReader(dataPath,fileName, 'smv')
    ImageIO.imWriter(outputPath,'5039.tif',im,3)'''

    ################## single tiff slice preprocessing ########################
    '''dataPath = 'C:/Tomosynthesis/localtest/'
    fileName = 'test-crop.tif'
    im = ImageIO.imReader(dataPath,fileName, 'tif',2)

    ## Equalization
    eqimg = histEqualization.histEqualization(im.data[0], 16)
    tiffLib.imsave(dataPath + 'test_eq.tif',eqimg)

    ## Denoising
    denoised = AT_denoising.DenoisingAW(im.data[0])
    tiffLib.imsave(dataPath + 'test_denoised.tif',denoised)

    ## Equlization + Denoising
    eq_denoised = AT_denoising.DenoisingAW(eqimg)
    tiffLib.imsave(dataPath + 'test_eq_denoised.tif',eq_denoised)

    ## Denoised + Equalization
    de_eq = histEqualization.histEqualization(denoised, 16)
    tiffLib.imsave(dataPath + 'test_de_eq.tif',de_eq)'''

    ############################ Gabor test #############################
    '''## Gabor kernel test
    dataPath = 'C:/Tomosynthesis/localtest/'
    outputPath = 'C:/Tomosynthesis/localtest/res/'
    fileName = 'test-crop.tif'
    im = ImageIO.imReader(dataPath,fileName, 'tif',2)

    kernels = gabor_filter.creat_Gabor_Kernels(4, 20, 0.0185,0.9)
    response = gabor_filter.compute_Response(im.data[0],kernels)

    gabor_filter.plot_Kernels(kernels)
    gabor_filter.plot_Response(response)

    for i in range(len(kernels)):
        tiffLib.imsave(outputPath + str(i) + 'kernels.tif',np.float32(kernels[i]))
        tiffLib.imsave(outputPath + str(i) + 'response.tif',np.float32(response[i]))'''

    ## Gabor filter bank test
    '''dataPath = 'C:/Tomosynthesis/localtest/'
    outputPath = 'C:/Tomosynthesis/localtest/res/'
    fileName = '7742_39-0026-2skeleton.tif'
    im = ImageIO.imReader(dataPath,fileName, 'tif',2)

    norietation = 4
    sigmas = (5,10,15,10)
    frequencies = (0.01,0,05,0.1)
    gammas = (1,1,5)

    filter_bank = gabor_filter.creat_FilterBank(norietation,sigmas,frequencies,gammas)
    responses = gabor_filter.compute_Responses(im.data[0],filter_bank)

    for i in range(len(filter_bank)):
        print i
        for j in range(len(filter_bank[i])):
            tiffLib.imsave(outputPath + str(i)+'_'+ str(j)+'_' + 'kernels.tif',np.float32(filter_bank[i][j]))
            tiffLib.imsave(outputPath + str(i)+'_'+ str(j)+'_' + 'response.tif',np.float32(responses[i][j]))'''
    

    ## Gabor kernel response analysis test
    dataPath = 'C:/Tomosynthesis/localtest/'
    outputPath = 'C:/Tomosynthesis/localtest/res/'
    fileName = 'test-crop.tif'
    im = ImageIO.imReader(dataPath,fileName, 'tif',2)

    params = []
    params.append([4, 20, 0.0185,0.9])
    #params.append([4, 5, 0.01,1])
    #params.append([4, 5, 0.025,1])
    #params.append([4, 5, 0.05,1])
    #params.append([4, 5, 0.075,1])

    #params.append([4, 10, 0.01,1])
    #params.append([4, 10, 0.025,1.7])
    #params.append([4, 10, 0.05,1])

    #params.append([4, 15, 0.01,1])
    #params.append([4, 15, 0.0175,1.5])
    #params.append([4, 15, 0.025,1])

    #params.append([4, 20, 0.01,1])
    #params.append([4, 20, 0.0175,1])
    #params.append([4, 20, 0.0175,1])
    
    for k in range(len(params)):
        print k
        sampRate = 30
        winSize = 15
        kernels = gabor_filter.creat_Gabor_Kernels(params[k][0],params[k][1],params[k][2],params[k][3])
        response = gabor_filter.compute_Response(im.data[0],kernels)
        (batchResp, integratedResp) = ra.cerat_batch_response(response,sampRate,winSize)
        poll = ra.vote(batchResp)
        integrated_poll = ra.integrating_poll(poll,sampRate,winSize,response[0].shape)
        
        tiffLib.imsave(outputPath + str(k) + 'poll.tif',np.float32(poll))
        tiffLib.imsave(outputPath + str(k) + 'integrated_poll.tif',np.float32(integrated_poll))
        
        '''for i in range(len(response)):                         
            tiffLib.imsave(outputPath + str(k) + '_' + str(i) + 'kernels.tif',np.float32(kernels[i]))
            tiffLib.imsave(outputPath + str(k) + '_' + str(i) + 'response.tif',np.float32(response[i]))
            tiffLib.imsave(outputPath + str(k) + '_' + str(i) + 'batchResp.tif',np.float32(batchResp[i]))
            tiffLib.imsave(outputPath + str(k) + '_' + str(i) + 'integratedResp.tif',np.float32(integratedResp[i]))'''
        patches = fex.patch_Extraction(im.data[0],poll,0,sampRate,90,7.5)
        ring = patches[0].getRing(10,60)
        
        for i in range(len(patches)):
            tiffLib.imsave(outputPath + str(i) + 'patches.tif',np.float32(patches[i].pdata))

main()
