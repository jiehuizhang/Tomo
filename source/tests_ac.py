import sys
import pymaxflow
import pylab
import numpy as np
from scipy import misc
import tiffLib
import histEqualization
import AT_denoising

import ImageIO
import TImage
import morphsnakes
import activeContourSegmentation as acSeg

import numpy as np
from scipy.misc import imread
from matplotlib import pyplot as ppl



def circle_levelset(shape, center, sqradius, scalerow=1.0):
    """Build a binary function with a circle as the 0.5-levelset."""
    R, C = np.mgrid[:shape[0], :shape[1]]
    phi = sqradius - (np.sqrt(scalerow*(R-center[0])**2 + (C-center[1])**2))
    u = np.float_(phi>0)
    return u

def test_mass():

    dataPath = 'C:/Tomosynthesis/localtest/'
    fileName = '5131R-recon08_45-1.tif'
    outputPath = 'C:/Tomosynthesis/localtest/res/'

    im = ImageIO.imReader(dataPath,fileName, 'tif',2)
    
    # padding borarders
    '''
    paddingrd = 10
    bordares = ((paddingrd,paddingrd),(paddingrd,paddingrd))
    paddingv = 10000
    bordarevs = ((paddingv,paddingv),(paddingv,paddingv))
    im = np.lib.pad(im.data[0], bordares, 'constant',constant_values = bordarevs)
    '''
    
    eqimg = histEqualization.histEqualization(im.data[0], 16)
    denoised = AT_denoising.DenoisingAW(eqimg)
    denoised = AT_denoising.DenoisingAW(denoised)
    denoised = AT_denoising.DenoisingAW(denoised)
    img = AT_denoising.DenoisingAW(denoised)
    tiffLib.imsave(outputPath + 'denoised.tif',img)

    # g(I)
    gI = morphsnakes.gborders(img, alpha=1, sigma=8)
    tiffLib.imsave(outputPath + 'gI.tif',np.float32(gI))
    # Morphological GAC. Initialization of the level-set.
    mgac = morphsnakes.MorphGAC(gI, smoothing=2, threshold=0.035, balloon=-1)
    mgac.levelset = circle_levelset(img.shape, (img.shape[0]/2, img.shape[1]/2), 140, scalerow=0.75)
    
    # Visual evolution.
    ppl.figure()
    ls = morphsnakes.evolve_visual(mgac, num_iters=110, background=img)
    tiffLib.imsave(outputPath + 'ls.tif',np.float32(ls))

def test_mass_2():

    dataPath = 'C:/Tomosynthesis/localtest/'
    fileName = '5131R-recon08_45-1.tif'
    outputPath = 'C:/Tomosynthesis/localtest/res/'

    im = ImageIO.imReader(dataPath,fileName, 'tif',2)
    
    # padding borarders
    '''
    paddingrd = 10
    bordares = ((paddingrd,paddingrd),(paddingrd,paddingrd))
    paddingv = 10000
    bordarevs = ((paddingv,paddingv),(paddingv,paddingv))
    im = np.lib.pad(im.data[0], bordares, 'constant',constant_values = bordarevs)
    '''
    
    eqimg = histEqualization.histEqualization(im.data[0], 16)
    denoised = AT_denoising.DenoisingAW(eqimg)
    denoised = AT_denoising.DenoisingAW(denoised)
    denoised = AT_denoising.DenoisingAW(denoised)
    img = AT_denoising.DenoisingAW(denoised)

    img = np.max(img) - img
    tiffLib.imsave(outputPath + 'denoised.tif',img)

    # g(I)
    gI = morphsnakes.gborders(img, alpha=1, sigma=8)
    tiffLib.imsave(outputPath + 'gI.tif',np.float32(gI))
    # Morphological GAC. Initialization of the level-set.
    mgac = morphsnakes.MorphGAC(gI, smoothing=2, threshold=0.035, balloon=1)
    mgac.levelset = circle_levelset(img.shape, (img.shape[0]/2, img.shape[1]/2), 15)
    
    # Visual evolution.
    ppl.figure()
    ls = morphsnakes.evolve_visual(mgac, num_iters=110, background=img)
    tiffLib.imsave(outputPath + 'ls.tif',np.float32(ls))

if __name__ == '__main__':

    dataPath = 'C:/Tomosynthesis/localtest/'
    fileName = 'cancer.tif'
    outputPath = 'C:/Tomosynthesis/localtest/res/'

    im = ImageIO.imReader(dataPath,fileName, 'tif',2)

    tiffLib.imsave(outputPath + 'image.tif',np.float32(im.data[0]))   #########

    eqimg = histEqualization.histEqualization(im.data[0], 16)
    denoised = AT_denoising.DenoisingAW(eqimg)
    denoised = AT_denoising.DenoisingAW(denoised)
    denoised = AT_denoising.DenoisingAW(denoised)
    imdata = AT_denoising.DenoisingAW(denoised)

    tiffLib.imsave(outputPath + 'preprocessed.tif',np.float32(imdata))   #########

    acSeg.ac_outwards(imdata)
    #acSeg.ac_inwards(imdata)
    
    
