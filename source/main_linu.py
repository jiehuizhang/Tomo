import time
from multiprocessing import Pool
import gc

import numpy as np
import scipy.ndimage.filters as filters
from pylab import *
import pickle

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
import MC_Detection as mc
import Creat_trainSam
import mass3Dextraction as mass3d
import Dimreduction
import classification
import graph_classification as grc
import activeContourSegmentation as acSeg
import morphsnakes

def f(x,y):
    z = math.factorial(200)
    return x*y

def f_wrapper(args):
    
    return f(*args)

if __name__ == '__main__':

    ############################## single smv convert #######################
    '''
    dataPath = '/home/TomosynthesisData/Cancer_25_cases/5092/'
    outputPath = '/home/yanbin/Tomosynthesis/data/5092/'
    fileName = '5092Recon08.smv_AutoCrop.smv'
    im = ImageIO.imReader(dataPath,fileName, 'smv')
    ImageIO.imWriter(outputPath,'5092.tif',im,3)
    '''

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
    '''dataPath = 'C:/Tomosynthesis/localtest/'
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
        sampRate = 30
        winSize = 15
        kernels = gabor_filter.creat_Gabor_Kernels(params[k][0],params[k][1],params[k][2],params[k][3])
        response = gabor_filter.compute_Response(im.data[0],kernels)
        (batchResp, integratedResp) = ra.cerat_batch_response(response,sampRate,winSize)
        poll = ra.vote(batchResp)
        integrated_poll = ra.integrating_poll(poll,sampRate,winSize,response[0].shape)
        
        tiffLib.imsave(outputPath + str(k) + 'poll.tif',np.float32(poll))
        tiffLib.imsave(outputPath + str(k) + 'integrated_poll.tif',np.float32(integrated_poll))
        
        for i in range(len(response)):                         
            tiffLib.imsave(outputPath + str(k) + '_' + str(i) + 'kernels.tif',np.float32(kernels[i]))
            tiffLib.imsave(outputPath + str(k) + '_' + str(i) + 'response.tif',np.float32(response[i]))
            tiffLib.imsave(outputPath + str(k) + '_' + str(i) + 'batchResp.tif',np.float32(batchResp[i]))
            tiffLib.imsave(outputPath + str(k) + '_' + str(i) + 'integratedResp.tif',np.float32(integratedResp[i]))
        patches = fex.patch_Extraction(im.data[0],poll,0,sampRate,90,7.5)
        patches_feats = np.zeros((1,10), dtype=np.double)
        for i in range(len(patches)):
            patches[i].getRings(numrings = 5)
            patches[i].getMeanFeats()
            patches[i].getVarFeats()
            patches_feats = np.vstack((patches_feats,patches[i].dumpFeats()))
            
        np.savetxt(outputPath + 'patches_feats.txt', patches_feats, delimiter='\t')   
        for i in range(len(patches)):
            tiffLib.imsave(outputPath + str(i) + 'patches.tif',np.float32(patches[i].pdata))'''

    ############################# Mass 3D Extraction sequentialy ########################
    '''
    dataPath = 'C:/Tomosynthesis/localtest/'
    fileName = 'test-crop.tif' 
    im = ImageIO.imReader(dataPath,fileName, 'tif',3)
    
    mass3d.Mass3dExtra(im)
    '''
    ############################ LOG test #############################
    # 2d test
    '''dataPath = 'C:/Tomosynthesis/localtest/'
    outputPath = 'C:/Tomosynthesis/localtest/res/'
    #fileName = '5016EMML08_17.tif'
    fileName = 'MC_1_5092Recon08_16-1.tif'
    im = ImageIO.imReader(dataPath,fileName, 'tif',2) 
    
    log = mc.log_filtering(im.data[0],winSize=40,sigma=3,fg_thresh = 0.6)
    constrained_log = mc.laebl_connecte_comp(log,threshold=3.0,size_constrain = (2,80))
    mcList = mc.MC_buildup_2d(im.data[0],constrained_log)
    mc.MC_connect_2d(mcList,dis_threshold = 300)
    for i in range(len(mcList)):
        print mcList[i].density_2d

    
    tiffLib.imsave(outputPath + 'log_constrained.tif',np.float32(constrained_log))
    tiffLib.imsave(outputPath + 'log_tile.tif',np.float32(log))'''

    # 3d test
    '''dataPath = 'C:/Tomosynthesis/localtest/'
    outputPath = 'C:/Tomosynthesis/localtest/res/5092/'
    fileName = '5092-2.tif'
    im = ImageIO.imReader(dataPath,fileName, 'tif',3)
    
    start = time.clock()
    mc_Lists = []
    for i in range(im.size_2):
        log = mc.log_filtering(im.data[i],winSize=40,sigma=3,fg_thresh = 0.6)
        constrained_log = mc.laebl_connecte_comp(log,threshold=3.0,size_constrain = (2,80))
        mcList = mc.MC_buildup_2d(im.data[i],constrained_log)
        mc.MC_connect_2d(mcList,dis_threshold = 300)
        for mc_item in mcList:
            mc_item.center[2] = i
        mc_Lists.append(mcList)
        tiffLib.imsave(outputPath + str(i) + 'log_constrained.tif',np.float32(constrained_log))
    end = time.clock()
    print end - start
    
    global_id = mc.MC_connect_3d(mc_Lists)
    gloabal_list = mc.MCs_constuct_3d(mc_Lists,global_id)
    MC_List_3D = mc.MCs_constrain(gloabal_list)

    for item in MC_List_3D:
        print(item.center, item.intensity, item.volume)'''
    
    # 3d parallel test
    '''dataPath = 'C:/Tomosynthesis/localtest/'
    outputPath = 'C:/Tomosynthesis/localtest/res/5092/'
    fileName = '5092-1.tif'
    im = ImageIO.imReader(dataPath,fileName, 'tif',3)
    print 'Finished Loading!'

    start = time.clock()
    mc_Lists = []   
    pool = Pool(processes=3)
    params =[(i,im.data[i]) for i in range(im.size_2)]
    mc_Lists = pool.map(mc.parallelWrapper,params)
    end = time.clock()
    print end - start
    
    global_id = mc.MC_connect_3d(mc_Lists)
    gloabal_list = mc.MCs_constuct_3d(mc_Lists,global_id)
    MC_List_3D = mc.MCs_constrain(gloabal_list)

    for item in MC_List_3D:
        print(item.center, item.intensity, item.volume)  '''

    ############################ Creat Training Sample #############################
    '''
    dataPath = 'C:/Tomosynthesis/training/cancer/'
    outputPath = 'C:/Tomosynthesis/localtest/res/'

    
    ## Rings intensity features 
    rings_feats = Creat_trainSam.creatTrainigSam(dataPath,opt = 'Rings', numrings = 8)
    np.savetxt(outputPath + 'rings_feats_cancer.txt', rings_feats, delimiter='\t')
    
    ## FD featues
    FD_feats = Creat_trainSam.creatTrainigSam(dataPath,opt = 'FD')
    np.savetxt(outputPath + 'FD_feats_cancer.txt', FD_feats, delimiter='\t')

    ## HOG features
    hog_feats = Creat_trainSam.creatTrainigSam(dataPath,opt = 'HOG')
    np.savetxt(outputPath + 'hog_feats_cancer.txt', hog_feats, delimiter='\t')

    feats = np.hstack((rings_feats,FD_feats,hog_feats))
    np.savetxt(outputPath + 'feats_cancer.txt', feats, delimiter='\t')
    '''


    ############################ HOG test #############################
    '''
    dataPath = 'C:/Tomosynthesis/localtest/'
    outputPath = 'C:/Tomosynthesis/localtest/res/'
    #fileName = 'test-crop-1.tif'
    fileName = '5131R-recon08_45-1.tif'
    im = ImageIO.imReader(dataPath,fileName, 'tif',2)

    # downsampling
    im.downSample(rate = 2)
    
    # histogram equalization
    eqimg = histEqualization.histEqualization(im.sampled_datas[0], 16)
    
    # smoothing
    smoothimg = filters.gaussian_filter(eqimg, sigma = 2, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0)
    
    patch = TPatch.TPatch()
    patch.initialize(smoothimg)
    patch.computeGradient()
    patch.gradOrieNormalize(threshold = 1500)
    patch.getGSectors(12)
    patch.getNormPerc(norm_th = 135)
    patch.getNormLevl()

    # plot
    t = range(patch.normal_percentage.shape[1])
    s = patch.normal_percentage[0]
    s2 = patch.normal_level[0]

    figure(1)
    subplot(211)
    plot(t, s)
    grid()
    
    subplot(212)
    plot(t, s2)
    title(fileName)
    grid()
    show()

    # save 
    f = open(outputPath + 'gsectors.txt', 'w')
    for item in patch.gsectors:
        f.write("%s\n" % item)
    f.close()
    #np.savetxt(outputPath + 'gsectors.txt', np.asarray(patch.gsectors), delimiter='\t')
     
    tiffLib.imsave(outputPath + fileName[0:11] + 'down_sampled.tif',np.float32(smoothimg))
    tiffLib.imsave(outputPath + fileName[0:11] +'gradient_magnitude.tif',np.float32(patch.gmagnitude))
    tiffLib.imsave(outputPath + fileName[0:11] +'gradient_orientation.tif',np.float32(patch.gorientation))
    tiffLib.imsave(outputPath + fileName[0:11] +'gradient_orientation_normalized.tif',np.float32(patch.gnormorientation))
    tiffLib.imsave(outputPath + fileName[0:11] +'gy.tif',np.float32(patch.gy))
    tiffLib.imsave(outputPath + fileName[0:11] +'local_orientation.tif',np.float32(patch.location_ori))
    tiffLib.imsave(outputPath + fileName[0:11] +'reflected_orientation.tif',np.float32(patch.greflorientation))
    '''

    ############################ FD Test #############################

    '''
    dataPath = 'C:/Tomosynthesis/training/cancer/'
    outputPath = 'C:/Tomosynthesis/localtest/res/'   
    fileName = '5047Recon08_51-1.tif'

    im = ImageIO.imReader(dataPath,fileName, 'tif',2)
    denoised = AT_denoising.DenoisingAW(im.data[0])
    eqimg = histEqualization.histEqualization(denoised, 16)

    patch = TPatch.TPatch()
    patch.initialize(eqimg)
    FD = patch.getFD()
    print FD
    tiffLib.imsave(outputPath + 'logpolarPSD.tif',np.float32(np.log10(patch.PSD_polar)))
    '''

    ############################ Dim reduction test #############################
    '''
    dataPath = 'C:/Tomosynthesis/localtest/res/'
    outputPath = 'C:/Tomosynthesis/localtest/res/'
    control_name = 'feats_control.txt'
    cancer_name = 'feats_cancer.txt'

    control = np.loadtxt(dataPath + control_name)
    cancer = np.loadtxt(dataPath + cancer_name)

    data = np.vstack((control,cancer))
    label = np.zeros((control.shape[0] + cancer.shape[0],),np.int)
    label[0:control.shape[0]-1] = 1

    optList = ['rand','pca','lda','iso','lle','mlle','HLLE','LTSA','randtree','spectral']
    pred = []
    accu = []

    for opt in optList:
       
        data_projected = Dimreduction.dim_Reduction(data, label, opt, n_components=2, visualize = True)
        classifier = classification.classifier(data_projected,label)
        classifier.train(opt ='SVM')
        classifier.classify()
    '''

    ############################# Mass 3D extraction ########################################
      
    dataPath = '/home/yanbin/Tomosynthesis/data/tiffs_3d/5016/'
    paraPath = '/home/yanbin/localtest/'
    outputPath = '/home/yanbin/Tomosynthesis/results/5016/'
    im_name = '5016EMML08.tif'
    

    # loading 
    im = ImageIO.imReader(dataPath,im_name, 'tif',3)
    print (im.size_0,im.size_1,im.size_2)
    
    control_name = 'feats_control_1.txt'
    cancer_name = 'feats_cancer.txt'
    control = np.loadtxt(paraPath + control_name)
    cancer = np.loadtxt(paraPath + cancer_name)

    # training
    data = np.vstack((control,cancer))
    label = np.zeros((control.shape[0] + cancer.shape[0],),np.int)
    label[0:control.shape[0]] = 1
    
    data_projected = Dimreduction.dim_Reduction(data, label, opt='randtree', n_components=5, visualize = False)
    classifier = classification.classifier(data_projected,label)
    classifier.train(opt ='SVM')


    # paralel
    start = time.clock()   
    pool = Pool(processes=6)
    #params =[(i,im.data[i],classifier) for i in range(im.size_2)]
    params =[(i,im.data[i],classifier) for i in range(65,69)]
    
    sliceList = []   
    sliceList = pool.map(mass3d.parallelWrapper,params)
    
    end = time.clock()
    print end - start

    # save the workspace
    output = open(outputPath + '5016.pkl', 'wb')
    pickle.dump(sliceList, output)
    output.close()
    

    ############################# Mass 3D extraction  Connecting ########################################
    '''
    featPath = 'C:/Tomosynthesis/localtest/res/'   
    pkl_file = open(path + 'workspace.pkl', 'rb')
    im = pickle.load(pkl_file)
    pkl_file.close()
    print im.size_2
    '''
        

    

