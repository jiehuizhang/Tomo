"""This script shows how to run the preprocessing procedure for a tiff stack."""
import time
from multiprocessing import Pool
import gc

import numpy as np
import scipy.ndimage.filters as filters
#from pylab import *
import pickle
from skimage.filter import threshold_otsu
from skimage.filter import threshold_adaptive
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from skimage.morphology import disk
from scipy import ndimage
from skimage.filter import roberts, sobel
import cv2.cv as cv

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
import midt
import PMHoughT
import TPSpline
import registration
import TPS_wrapper
import regionCompairision as rC

if __name__ == '__main__':

    dataPath = 'C:/Tomosynthesis/localtest/'
    outputPath = 'C:/Tomosynthesis/test_script/'
    fileName = '5062fullrecon108-1_skin_remove.tif'

    im = ImageIO.imReader(dataPath,fileName, 'tif')
    print (im.size_0,im.size_1,im.size_2)

    selem = disk(15)
    for i in range(im.size_2):

        # skin-line remove
        threshold = 8000
        mask = im.data[i] > threshold
        
        sline = np.zeros(im.data[i].shape, int)
        sline[mask] = 1
        
        dilated = ndimage.convolve(sline, selem, mode='constant', cval=0)

        mask = dilated > 0
        im.data[i][mask] = 0


        # equalization
        # eqimg = histEqualization.histEqualization(im.data[i], 16)
        # tiffLib.imsave(outputPath + str(i) + 'eq.tif',eqimg)'
