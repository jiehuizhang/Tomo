import numpy as np

import ImageIO
import TImage
import ShapeIndex
import histEqualization
import AT_denoising
from scipy import misc
import tiffLib
import gabor_filter

def main():

    '''# singla smv convert
    dataPath = '/home/TomosynthesisData/Cancer_25_cases/5039/'
    outputPath = '/home/yanbin/Tomosynthesis/processed/5039/'
    fileName = '5039.recon08.smv'
    im = ImageIO.imReader(dataPath,fileName, 'smv')
    ImageIO.imWriter(outputPath,'5039.tif',im,3)

    # singal tiff slice preprocessing
    dataPath = '/home/yanbin/Tomosynthesis/test/'
    fileName = 'test.tif'
    im = ImageIO.imReader(dataPath,fileName, 'tif',2)

    # Equalization
    eqimg = histEqualization.histEqualization(im.data[0], 16)
    tiffLib.imsave(dataPath + 'test_eq.tif',eqimg)

    # Denoising
    denoised = AT_denoising.DenoisingAW(im.data[0])
    tiffLib.imsave(dataPath + 'test_denoised.tif',denoised)

    # Equlization + Denoising
    eq_denoised = AT_denoising.DenoisingAW(eqimg)
    tiffLib.imsave(dataPath + 'test_eq_denoised.tif',eq_denoised)

    # Denoised + Equalization
    de_eq = histEqualization.histEqualization(denoised, 16)
    tiffLib.imsave(dataPath + 'test_de_eq.tif',de_eq)'''

    # Gabor filter testing
    dataPath = 'C:/Tomosynthesis/localtest/'
    fileName = 'test-crop.tif'
    im = ImageIO.imReader(dataPath,fileName, 'tif',2)

    kernels = gabor_filter.creat_Gabor_Kernels(8, 20, 0.01,1.5)
    response = gabor_filter.compute_Response(im.data[0],kernels)
    gabor_filter.plot_Kernels(kernels)
    gabor_filter.plot_Response(response)

    for i in range(len(response)):
        tiffLib.imsave(dataPath + str(i) + 'kernels.tif',np.float32(kernels[i]))
        tiffLib.imsave(dataPath + str(i) + 'response.tif',np.float32(response[i]))

    print 'done'
    
main()
