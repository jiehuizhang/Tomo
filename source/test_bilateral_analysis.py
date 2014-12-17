"""This script shows how to run registration and comparision.
All examples in this script are tested on 2D image slices."""
import TPSpline
import registration
import TPS_wrapper
import regionCompairision as rC

import ImageIO
import TImage
import tiffLib

import numpy as np
def test_func():
    """Please set flag Registration to 1 if you want to run registration.
    There are two parameters need to specify:
    The third one specifys how many pairs of fiducial points.
    The forth prameter specifys if you want to run in python(slow) or c++.

    Please set RegionComparison to 1 if you want to run comparision
    Make sure you have the two registered images in the data directory.
    Parameters includes region size, set to 200 by default.
    The output is a distance image in which pixel values are distance of the region.
    """

    dataPath = 'C:/Tomosynthesis/localtest/reg/'
    outputPath = 'C:/Tomosynthesis/test_script/'
   
    fileName_r = '6044_r.tif'
    fileName_l = '6044_l.tif'

    im_r = ImageIO.imReader(dataPath,fileName_r, 'tif',2)
    im_l = ImageIO.imReader(dataPath,fileName_l, 'tif',2)

    ## Run flags
    Registration = 1
    RegionComparison = 1
    
    ## Run registration
    if Registration == 1:

        warped_im1 = registration.registration(im_r.data[0], im_l.data[0], 15,'py', outputPath)
        if warped_im1 != None:
            tiffLib.imsave(outputPath + 'dst.tif',np.float32(warped_im1))

    ## Run region comparison
    if RegionComparison == 1:

        params = []
        params.append(('1d', 'cv_comp', cv.CV_COMP_CORREL))
        params.append(('1d', 'scipy_comp', 'Euclidean'))
        params.append(('1d', 'scipy_comp', 'Manhattan'))
        params.append(('1d', 'kl_div', 'None'))

        params.append(('2d', cv.CV_TM_SQDIFF_NORMED, 'None'))
        params.append(('2d', cv.CV_TM_CCORR_NORMED, 'None'))
        params.append(('2d', cv.CV_TM_CCOEFF_NORMED, 'None'))

        params.append(('decomp', 'eigen', 'None'))
        params.append(('decomp', 'NMF', 'None'))


        for i in range(len(params)):
            dis_im = rC.imageComp(im_r.data[0],im_l.data[0], params[i], region_s = 200, olap_s = 200)   
            tiffLib.imsave(outputPath + 'dis' + str(i) + '.tif' ,np.float32(dis_im) )


if __name__ == '__main__':

    test_func()
