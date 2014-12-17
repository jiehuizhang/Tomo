"""Compare regions in two sides images based on variety of metrics"""

import numpy as np
from scipy.spatial import distance as dist
from sklearn.decomposition import ProjectedGradientNMF
from scipy import linalg as LA
import matplotlib.pyplot as plt
import numpy as np
import argparse
import glob
import cv2
import cv2.cv as cv


def exHistgram(imregion, lb = 2000, up = 8000,s_bin = 100, normalize = True):
    """Extract histogram from a image region.

    Parameters
    ----------
    imregion : numpy array (2D)
        The image region data
    lb : integer
        Lower bound of the histogram
    ub : integer
        Upper bound of the histogram
    s_bin:
        The bin size of the histogram
    normalize: bool
        If or not to normalize the histogram
        
    """

    ## create bins with input lower bound, upper bound and bin size
    bins = range(lb,up,s_bin)

    ## convert matrix into histogram
    histgram = np.histogram(imregion.reshape(imregion.size), bins)
    
    # normalize
    if normalize:
        histgram = cv2.normalize(histgram[0]).flatten()

    return histgram


def compHist(hist1, hist2, method, formula):
    """Compare two histograms with given method and formula.

    Parameters
    ----------
    hist1 : 1D array
        The first histogram
    hist2 : 1D array
        The second histogram
    method : str(cv integer)
        Options for method ('cv_comp', 'scipy_comp', 'kl_div')
    formula: str(cv integer)
        Options for formula.
        For method == 'cv_comp' (cv.CV_COMP_CORREL, cv.CV_COMP_CHISQR, cv.CV_COMP_INTERSECT, cv.CV_COMP_BHATTACHARYYA)
        For method == 'scipy_comp' ("Euclidean", "Manhattan", "Chebysev")
        
    """

    ## using opencv
    if method == 'cv_comp':
        dis = cv2.compareHist(np.float32(hist1), np.float32(hist2), formula)
        if formula == cv.CV_COMP_CORREL:
            dis = -dis + 1

    ## using Scipy distance metrics
    if method == 'scipy_comp':
        if formula == 'Euclidean':
            dis = dist.euclidean(hist1, hist2)
        if formula == 'Manhattan':
            dis = dist.cityblock(hist1, hist2)
        if formula == 'Chebysev':
            dis = dist.chebyshev(hist1, hist2)

    ## using KL divergence
    hist1 = np.float32(hist1) + 1
    hist2 = np.float32(hist2) + 1     
    if method == 'kl_div':
        kbp = np.sum(hist1 * np.log(hist1 / hist2), 0)
        kbq = np.sum(hist2 * np.log(hist2 / hist1), 0)

        dis = np.double(kbp + kbq)/2

    return dis


def tempMatch(imregion1,imregion2,method):
    """Compare two image regions with given method.

    Parameters
    ----------
    imregion1 : 2D array
        The image data of region 1
    imregion2 : 2D array
        The image data of region 2
    method : cv integer
        Options for method (cv.CV_TM_SQDIFF, cv.CV_TM_SQDIFF_NORMED, cv.CV_TM_CCORR,
        cv.CV_TM_CCORR_NORMED, cv. CV_TM_CCOEFF, CV_TM_CCOEFF_NORMED)
        
    """

    # Apply template Matching
    dis = cv2.matchTemplate(np.float32(imregion1),np.float32(imregion2),method)

    if method == cv.CV_TM_SQDIFF_NORMED or method == cv.CV_TM_CCOEFF_NORMED:
        dis = 1-dis

    return dis


def matdecomp(imregion, method):
    """Compute matrix decomposition

    Parameters
    ----------
    imregion : 2D array
        The image region data
    method : str
        Options for method ('eigen', 'NMF')
        
    """

    if method == 'eigen':
        ## columns are eigen vectors
        e_vals, e_vecs = LA.eig(imregion)

        return e_vecs

    if method == 'NMF':
        model = ProjectedGradientNMF(n_components=2, init='random',random_state=0)
        model.fit(imregion)
        
        comp = model.components_
        err = model.reconstruction_err_

        return comp
        

def regionComp(imregion1,imregion2, dis_opt, method,formula):
    """Main region comparision function

    Parameters
    ----------
    imregion1 : 2D array
        The image data of region 1
    imregion2 : 2D array
        The image data of region 2
    dis_opt: str
        Options for comparing dimensionality ('1d', '2d', 'decomp')
    method : str (cv integer)
        Options for method 
    formula: str (cv integer)
        Options for formula
        
    """

    ## 1-d distribution comparision    
    # extract distribution as histogram
    histg1 = exHistgram(imregion1)
    histg2 = exHistgram(imregion2)
    
    # compare histogram saimilarity
    if dis_opt == '1d':
        '''
        Options for method includes: 'cv_comp','scipy_comp','kl_div'.
        
        if method == 'cv_comp', options for formula includes:
            cv.CV_COMP_CORREL, cv.CV_COMP_CHISQR, cv.CV_COMP_INTERSECT, cv.CV_COMP_BHATTACHARYYA
            the first one yields continuals results

        if method == 'scipy_comp', options for formula includes:
            "Euclidean", "Manhattan", "Chebysev"
            the first two yield continuals results

        if method == 'kl_div', no options needed for formula:
            formula = 'None'
        '''
                
        dis_1d = compHist(histg1,histg2,method, formula)
            
        return dis_1d

    ## 2-d matrix comparision   
    # compare matrix correlation
    if dis_opt == '2d':
        '''
        Options for formula includes:
        cv.CV_TM_SQDIFF, cv.CV_TM_SQDIFF_NORMED, cv.CV_TM_CCORR, cv.CV_TM_CCORR_NORMED, cv. CV_TM_CCOEFF, CV_TM_CCOEFF_NORMED
        Only the normalized ones are used
        '''
        dis_2d = tempMatch(imregion1,imregion2,method)
        return dis_2d

    ## feature based comparision
    # matrix decomposition
    if dis_opt == 'decomp':

        if np.mean(imregion1) < 200:
            return 0
        '''
        Options for method includes:
        'eigen', 'NMF'
        '''
        vect1 = matdecomp(imregion1, method)
        vect2 = matdecomp(imregion2, method)

        # compute feature similarity
        dis_f = vectComp(vect1, vect2, numComp = 2)
        
        return dis_f

def vectComp(vecMat1, vecMat2, numComp = 2):
    """Compute distance between two vector matrices

    Parameters
    ----------
    vecMat1 : 
        The first vector matrix
    vecMat2 : 
        The second vector matrix
    numComp: integer
        Number of components remainded
        
    """

    dis = 0
    for i in range(numComp):
        dis = dis + compHist(vecMat1[:,i],vecMat2[:,i],method = 'scipy_comp', formula = "Euclidean")

    return dis
    


def imageComp(im1,im2, params, region_s = 200, olap_s = 200):
    """Main image comparision function

    Parameters
    ----------
    im1 : 2D array
        The image data of image 1
    im2 : 2D array
        The image data of image 2
    params: tuple
        Options of method and formula for comparing 
    region_s : integer
        Cropping region size
    olap_s: integer
        Overlapping size.
        
    """

    # construct result image
    r = min(im1.shape[0],im2.shape[0])
    c = min(im1.shape[1],im2.shape[1])
    dis_im = np.zeros((r,c), np.float32)

    # calculate number of regions
    n_rr = r/olap_s - 1
    n_cr = c/olap_s - 1

    for i in range(n_rr):
        for j in range(n_cr):
            imregion1 = im1[(i*olap_s) : (i*olap_s+region_s), (j*olap_s) : (j*olap_s+region_s)]
            imregion2 = im2[(i*olap_s) : (i*olap_s+region_s), (j*olap_s) : (j*olap_s+region_s)]
            dis_im[(i*olap_s) : (i*olap_s+region_s), (j*olap_s) : (j*olap_s+region_s)] = regionComp(imregion1,imregion2, params[0],params[1],params[2])

    return dis_im

    

    
    
    
    
    
        
            


        

