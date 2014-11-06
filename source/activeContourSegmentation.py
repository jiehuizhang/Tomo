'''AC segmentation'''

import sys
#import pymaxflow
import numpy as np
from scipy import misc
import tiffLib
import histEqualization
import AT_denoising
from scipy import stats
import cv2
import math

import ImageIO
import TImage
import morphsnakes
from scipy import ndimage
import matplotlib
#from matplotlib import pyplot as ppl
from numpy.linalg import lstsq

import skimage
from skimage.morphology import label
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import disk
from skimage.measure import regionprops
from skimage.measure import find_contours
import bsckground_Substraction as bs
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter, gaussian_gradient_magnitude

def circle_levelset(shape, center, sqradius, scalerow=1.0):
    """Build a binary function with a circle as the 0.5-levelset."""
    R, C = np.mgrid[:shape[0], :shape[1]]
    phi = sqradius - (np.sqrt(scalerow*(R-center[0])**2 + (C-center[1])**2))
    u = np.float_(phi>0)
    return u

def ac_inwards(imdata,visulization = False):
    
    #  calculate gradient information (I)
    gI = morphsnakes.gborders(imdata, alpha=0.5, sigma=2)
        
    # Morphological GAC. Initialization of the level-set.
    mgac = morphsnakes.MorphGAC(gI, smoothing=1, threshold=0.035, balloon=-1)
    mgac.levelset = circle_levelset(imdata.shape, (imdata.shape[0]/2, imdata.shape[1]/2), 110, scalerow=1.0)
    
    # Visual evolution.
    if visulization ==  True:
        matplotlib.pyplot.figure()
    ls = morphsnakes.evolve_visual(mgac,visulization, num_iters=80, background=imdata)

    return ls

def ac_outwards(imdata,visulization = False):

    imdata = np.max(imdata) - imdata

    # calculate gradient information (I)
    gI = morphsnakes.gborders(imdata, alpha=1.3, sigma=2)

    # Morphological GAC. Initialization of the level-set.
    mgac = morphsnakes.MorphGAC(gI, smoothing=2, threshold=0.028, balloon=1)
    mgac.levelset = circle_levelset(imdata.shape, (imdata.shape[0]/2, imdata.shape[1]/2), 15)
    
    # Visual evolution.
    if visulization ==  True:
        matplotlib.pyplot.figure()
    ls = morphsnakes.evolve_visual(mgac, visulization, num_iters=130, background=imdata)

    return ls

def getCentSkewness(labim,area,index,centroid):

    cen_skew = 0.0
    for i in range(labim.shape[0]):
        for j in range(labim.shape[1]):
            if labim[i,j] == index + 1:
                cen_skew = cen_skew + abs(i - centroid[0]) + abs(j - centroid[1])

    cen_skew = cen_skew/area

    return cen_skew

def getUniqueContour(contour):
    
    length = contour.shape[0]
    cont_list = []
    i = 0
    while i < length -1:
        cont_list.append(i)
        p = contour[i]
        for j in range(i+1,length):
            if p[0] == contour[j][0] and p[1] == contour[j][1]:
                doingnothing = 0
            else:
                i = j
                break
        if j == length -1:
            break

    uniqcontour = np.zeros((len(cont_list),2),np.int_)
    uniqcontour = contour[np.array(cont_list),:]

    return uniqcontour

def getRBSTim(labim,im):

    outputPath = 'C:/Tomosynthesis/localtest/res/'

    height = labim.shape[0] - 1
    width = labim.shape[1] - 1

    #get Contour
    contour = np.floor(find_contours(labim, 0.5, fully_connected='low', positive_orientation='low')[0])
    contour = np.int_(contour)
    contour = getUniqueContour(contour)

    slop_r = 3
    normal_r = 40
    length = contour.shape[0]
    radius = 2
    
    RBST = np.zeros((normal_r,length))
    RBSTim = np.zeros(labim.shape,np.double)
    contim = np.zeros(labim.shape,np.double)
    for i in range(length):
        # linear regression of the neighbourhood
        p = contour[i]
        if p[0] == 0 or p[0] == height or p[1] == 0 or p[1] == width:
            continue
        contim[p[0],p[1]] = i
        p1 = np.double(contour[(i - slop_r + length)%length] - p)   
        p2 = np.double(contour[(i + slop_r)%length] - p)

        if p1[0] == p2[0]:
            if labim[p[0]+1,p[1]] == 1:
                for t in range(-normal_r, 0):
                    j = -normal_r - t
                    cord_x = p[0] + j
                    cord_y = p[1]
                    if cord_x < 0 or cord_y < 0 or cord_x > height or cord_y > width :
                        break
                    if labim[cord_x,cord_y] == 1:
                        break
                    RBSTim[cord_x,cord_y] = 1
                    bound_x = np.arange(max(cord_x-radius,0),min(cord_x+radius,height))                    
                    bound_y = np.arange(max(cord_y-radius,0),min(cord_y+radius,width))
                    RBST[abs(j),i] = np.mean(im[bound_x,:][:,bound_y])
            else:
                for j in range(1,normal_r):
                    cord_x = p[0] + j
                    cord_y = p[1]
                    if cord_x < 0 or cord_y < 0 or cord_x > height or cord_y > width :
                        break
                    if labim[cord_x,cord_y] == 1:
                        break
                    RBSTim[cord_x,cord_y] = 1
                    bound_x = np.arange(max(cord_x-radius,0),min(cord_x+radius,height))                    
                    bound_y = np.arange(max(cord_y-radius,0),min(cord_y+radius,width))
                    RBST[j ,i] = np.mean(im[bound_x,:][:,bound_y])
        if p1[1] == p2[1]:
            if labim[p[0],p[1] + 1] == 1:
                for t in range(-normal_r, 0):
                    j = -normal_r - t
                    cord_x = p[0]
                    cord_y = p[1] + j
                    if cord_x < 0 or cord_y < 0 or cord_x > height or cord_y > width :
                        break
                    if labim[cord_x,cord_y] == 1:
                        break
                    RBSTim[cord_x,cord_y] = 1
                    bound_x = np.arange(max(cord_x-radius,0),min(cord_x+radius,height))                    
                    bound_y = np.arange(max(cord_y-radius,0),min(cord_y+radius,width))
                    RBST[abs(j) ,i] = np.mean(im[bound_x,:][:,bound_y])
                    
            else:
                for j in range(1,normal_r):
                    cord_x = p[0]
                    cord_y = p[1] + j
                    if cord_x < 0 or cord_y < 0 or cord_x > height or cord_y > width :
                        break
                    if labim[cord_x,cord_y] == 1:
                        break
                    RBSTim[cord_x,cord_y] = 1
                    bound_x = np.arange(max(cord_x-radius,0),min(cord_x+radius,height))                    
                    bound_y = np.arange(max(cord_y-radius,0),min(cord_y+radius,width))
                    RBST[j ,i] = np.mean(im[bound_x,:][:,bound_y])           
        if p1[0] != p2[0] and p1[1] != p2[1]:
            gr = np.double(p2[0] - p1[0])/np.double(p2[1] - p1[1])
            if abs(gr)<1:              
                for t in range(-70,0):
                    x = -70 - t
                    y = gr*x
                    dist = math.sqrt(y**2+x**2)
                    cord_x = p[0]+ int(x)
                    cord_y = p[1] - int(y)
                    if cord_x < 0 or cord_y < 0 or cord_x > height or cord_y > width :
                        break
                    if dist < normal_r:
                        if labim[cord_x,cord_y] == 1:
                            break
                        RBSTim[cord_x,cord_y] = 1
                        bound_x = np.arange(max(cord_x-radius,0),min(cord_x+radius,height))                    
                        bound_y = np.arange(max(cord_y-radius,0),min(cord_y+radius,width))
                        RBST[int(dist) ,i] = np.mean(im[bound_x,:][:,bound_y])                                        
                for x in range(1,70):
                    y = gr*x
                    dist = math.sqrt(y**2+x**2)
                    cord_x = p[0]+ int(x)
                    cord_y = p[1] - int(y)
                    if cord_x < 0 or cord_y < 0 or cord_x > height or cord_y > width :
                        break
                    if dist < normal_r:
                        if labim[cord_x,cord_y]==1:
                            break
                        RBSTim[cord_x,cord_y] = 1
                        bound_x = np.arange(max(cord_x-radius,0),min(cord_x+radius,height))                    
                        bound_y = np.arange(max(cord_y-radius,0),min(cord_y+radius,width))
                        RBST[int(dist) ,i] = np.mean(im[bound_x,:][:,bound_y])
            else:
                for x in range(-70,0):
                    y = x/gr
                    dist = math.sqrt(y**2+x**2)
                    cord_x = p[0] + int(y)
                    cord_y = p[1] - int(x)
                    if cord_x < 0 or cord_y < 0 or cord_x > height or cord_y > width :
                        continue
                    if dist < normal_r:
                        if labim[cord_x,cord_y] == 1:
                            break
                        RBSTim[cord_x,cord_y] = 1
                        bound_x = np.arange(max(cord_x-radius,0),min(cord_x+radius,height))                    
                        bound_y = np.arange(max(cord_y-radius,0),min(cord_y+radius,width))                          
                        RBST[int(dist) ,i] = np.mean(im[bound_x,:][:,bound_y])
                for x in range(1,70):
                    y = x/gr
                    dist = math.sqrt(y**2+x**2)
                    cord_x = p[0] + int(y)
                    cord_y = p[1] - int(x)
                    if cord_x < 0 or cord_y < 0 or cord_x > height or cord_y > width:
                        break
                    if dist < normal_r:
                        if labim[cord_x,cord_y]==1:
                            break
                        RBSTim[cord_x,cord_y] = 1
                        bound_x = np.arange(max(cord_x-radius,0),min(cord_x+radius,height))                    
                        bound_y = np.arange(max(cord_y-radius,0),min(cord_y+radius,width))
                        RBST[int(dist) ,i] = np.mean(im[bound_x,:][:,bound_y])

    for i in range(1,normal_r):
        for j in range(1,length):
            if RBST[i,j] == 0:
                RBST[i,j] = RBST[i-1,j]
            if RBST[i,j] == 0:
                RBST[i,j] = RBST[i,j-1]
                
                
    RBSTgr = np.zeros(RBST.shape, np.double)
    gaussian_gradient_magnitude(RBST, sigma = 1.5, output=RBSTgr, mode='constant')
    h_gr = np.mean(RBSTgr,0)
    h_gr = np.tile(h_gr, (40, 1))

    thresh = 3200
    mask = h_gr>thresh
    h_gr_bin = np.zeros(h_gr.shape)
    h_gr_bin[mask] = 1

    ocross = 0
    for i in range(1,h_gr.shape[1]):
        if h_gr_bin[0,i] - h_gr_bin[0,i-1] == 1:
            ocross = ocross +1
    return ocross   
    '''          
    tiffLib.imsave(outputPath + 'RBSTim.tif',np.float32(RBSTim))   #########
    tiffLib.imsave(outputPath + 'RBST.tif',np.float32(RBST))   #########
    tiffLib.imsave(outputPath + 'contim.tif',np.float32(contim))   #########
    tiffLib.imsave(outputPath + 'RBSTgr.tif',np.float32(RBSTgr))   #########
    tiffLib.imsave(outputPath + 'h_gr.tif',np.float32(h_gr))   #########
    tiffLib.imsave(outputPath + 'h_gr_bin.tif',np.float32(h_gr_bin))   #########
    '''
    
def getLabelImFeats(lsim,center,orgim):
    
    label_img = skimage.measure.label(lsim)
    regions = regionprops(label_img)
    index = label_img[center[0],center[1]]-1

    # direct features
    Area = regions[index].area
    CentralMoments = regions[index].moments_central
    Eccentricity = regions[index].eccentricity
    Perimeter = regions[index].perimeter

    skewx=np.mean(stats.skew(lsim, axis=0, bias=True))
    skewy=np.mean(stats.skew(lsim, axis=1, bias=True))
    
    # derived features
    compact = Area/Perimeter**2
    skewness = np.sqrt(skewx**2 + skewy**2)
    cen_skew = getCentSkewness(label_img,Area, index,regions[index].centroid)
    numBranch = getRBSTim(label_img,orgim)

    return np.hstack((Area, Eccentricity, Perimeter, compact, skewness, cen_skew, numBranch))
