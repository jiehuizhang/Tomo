import skimage
from skimage import filter
from scipy import ndimage
from skimage.morphology import label, disk
from skimage.filter import roberts, sobel
from skimage.measure import find_contours
from skimage import measure
from skimage.morphology import disk
import numpy as np
import math
import tiffLib

import activeContourSegmentation as acSeg
import TPSpline
import TPS_wrapper

def registration(im1, im2, num = 10, opt = 'py'):

    # determin which one is the right side of the breast
    b_size = 5
    n_row, n_col = im1.shape
    side = 0
    if np.sum(im1[0:b_size,0:b_size]) < np.sum(im1[0:b_size,n_col-b_size:n_col]):
        side = 1   

    # flip the right side image
    if side == 1:
        im1 = np.fliplr(im1)
    else:
        im2 = np.fliplr(im2)           

    # find edges of both images
    edge1 = findEdge(im1)
    edge2 = findEdge(im2)

    # tune edges of both side
    edge1 = tuneEdge(edge1,im1.shape)
    edge2 = tuneEdge(edge2,im2.shape)

    # samping from both side
    points1 = contour_sampling(edge1, num)
    points2 = contour_sampling(edge2, num)

    # for debugging .........................
    sam_im1 = np.zeros(im1.shape,np.float32)
    for point in points1:
        sam_im1[point[0],point[1]] = 1

    sam_im2 = np.zeros(im2.shape,np.float32)
    for point in points2:
        sam_im2[point[0],point[1]] = 1
    
    selem = disk(15)
    dilated1 = ndimage.convolve(sam_im1, selem, mode='constant', cval=0)
    dilated2 = ndimage.convolve(sam_im2, selem, mode='constant', cval=0)

    
    oppath = 'C:/Tomosynthesis/localtest/reg/'
    points1 = np.asarray(points1)
    points2 = np.asarray(points2)
    np.savetxt(oppath + 'ps.txt', points1, '%d', delimiter=' ')   # X is an array
    np.savetxt(oppath + 'pd.txt', points2, '%d', delimiter=' ')   # X is an array
    tiffLib.imsave('C:/Tomosynthesis/localtest/reg/'+ 'im__1.tif',im1)
    
    # Thin Plate Spline interpolation
    dst = np.zeros(im1.shape)
    # im1 as source
    if opt == 'py':      
        tps = TPSpline.TPSpline()
        tps.setCorrespondences(points1, points2)
        dst = tps.warpImage(im1)
    '''
    if opt == 'c':
        dst = np.zeros(im1.shape)
        tps = TPS_wrapper.TPScpp(points1, points2)      
        tps.warp(im1, dst, lamda, interpolation, tpsInter)  
    '''
    return (dst, dilated1, dilated2) 

def findEdge(imdata):

    # remove border effect
    imdata[:,0] = imdata[:,1]
    
    # threshold segmentation
    #val = filter.threshold_otsu(imdata)
    val = 500
    mask = imdata >val

    contour = np.floor(find_contours(mask, 0.5, fully_connected='low', positive_orientation='low')[0])
    contour = np.int_(contour)
    contour = acSeg.getUniqueContour(contour)

    return contour

def tuneEdge(contour, imshape):

    r_contour = []
    upedge = 100
    leftedge = 30
    n_r, n_c = imshape
    
    for i in range(contour.shape[0] -1):

        # remove top effect
        if contour[i,0] < upedge:
            continue

        # remove bottom effect 
        if contour[i,1] < contour[i+1,1] and contour[i,0] > 3*n_r/4 and contour[i,1] < n_c/4:
            break
        
        else:
            r_contour.append(contour[i])

    return r_contour           

def contour_sampling(contour, num,delta = 5):
    
    samples = []
    step = len(contour)/num

    lefte = 0
     
    for i in range(num):
    #for i in range(5):
        samples.append(contour[i*step - i*delta])
        samples.append((contour[i*step - i*delta][0],lefte))

    #samples.append(contour[len(contour) - 1])

    return samples
               

def curvature(contour,fn = 3, bn = 3):

    clen = contour.shape[0]
    E = np.zeros((clen,), np.float32)
    thetai = np.zeros((clen,), np.float32)

    for k in range(1,clen):
    
        # first and last few points
        if k < bn:
            bnd = 0
            fnd = k + fn
        elif k + fn > clen-1:
            bnd = k - bn
            fnd = clen-1
        else:
            bnd = k - bn
            fnd = k + fn

        # calculate curvature
        lb = math.sqrt( (contour[k,0]-contour[bnd,0])**2 + (contour[k,1]-contour[bnd,1])**2 )
        lf = math.sqrt( (contour[k,0]-contour[fnd,0])**2 + (contour[k,1]-contour[fnd,1])**2 )

        if contour[k,1]-contour[bnd,1]!=0:
            thetab=math.atan( np.double(abs(contour[k,0]-contour[bnd,0])) / np.double(abs(contour[k,1]-contour[bnd,1])) )
        else:
            thetab=math.atan( np.double(abs(contour[k,0]-contour[bnd,0])) / np.double(abs(contour[k,1]-contour[bnd,1])) )
            thetab = math.pi/2 - thetab

        if contour[k,1]-contour[fnd,1]!=0:
            thetaf=math.atan( np.double(abs(contour[k,0]-contour[fnd,0])) / np.double(abs(contour[k,1]-contour[fnd,1])) )
        else:
            thetaf=math.atan( np.double(abs(contour[k,0]-contour[fnd,0])) / np.double(abs(contour[k,1]-contour[fnd,1])) )
            thetaf = math.pi/2 - thetaf

        thetai[k]=(thetab+thetaf)/2
        detlaf=abs(thetaf-thetai[k])
        detlab=abs(thetai[k]-thetab)
        E[k]=detlaf/lf/2+detlab/lb/2

    E[0]=E[1]
    E[clen - 1]=E[clen - 2]
    thetai[0]=thetai[1]
    thetai[clen - 1]=thetai[clen - 2]

    return (E,thetai)
                
            






        


