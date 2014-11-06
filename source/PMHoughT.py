""" Pectoral Muscle Detection """

from skimage.transform import (hough_line, hough_line_peaks, probabilistic_hough_line)
from skimage.filter import canny
from skimage import data
from scipy import ndimage
import scipy.ndimage.filters as filters

import numpy as np
import matplotlib.pyplot as plt

def line_triming(lines,side,n_row, n_col):
    
    t_lines = []
    dis_mid = []
    length = []
   
    for line in lines:
        p0, p1 = line
        if p0[1]- p1[1] == 0:
            continue
        slope = np.double(p0[0] - p1[0])/np.double(p0[1]- p1[1])
        mid = ((p0[0]+p1[0])/2,(p0[1]+p1[1])/2)
        if side == 1:
            quater = (n_col*3/4,n_row/4)
            if slope>0.5 and slope<2  and mid[0] > n_col/3 and mid[1] < n_row/3 :
                t_lines.append(line)
                dis_mid.append( np.sqrt((mid[0] - quater[0])**2 + (mid[1] - quater[1])**2) )
                length.append( np.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2) )
                
        if side == 0:
            quater = (n_col*1/4,n_row/4)
            if slope<-0.5  and slope>-2 and mid[0] < n_col/3 and mid[1] < n_row/3 :
                t_lines.append(line)
                dis_mid.append( np.sqrt((mid[0] - quater[0])**2 + (mid[1] - quater[1])**2) )
                length.append( np.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2) )
	
    if len(dis_mid) > 0:
        min_dis = np.argmin(dis_mid)
        return [t_lines[min_dis]]
    if len(length) > 0:
        max_len = np.argmax(length)
        
    
    return []

def mmask(ishape, line, side):
    
    p0,p1 = line
    slope = np.double(p0[1] - p1[1])/np.double(p0[0] - p1[0])
    intercept = slope*p0[0] - p0[1]

    X, Y = np.ogrid[0:ishape[0], 0:ishape[1]]
    mask = X - slope*Y + intercept < 0
    return mask


def PMremove(image,threshold = 15.5, visulization = False):

    # binarizing
    mask = image < threshold
    image[mask] = 0
    
    # smoothing
    smoothimg = filters.gaussian_filter(image, sigma = 1.0, order=0, output=None, mode='constant', cval=0.0, truncate=4.0)

    # Hough transform
    edges = canny(image, 3)
    lines = probabilistic_hough_line(smoothimg, threshold=10, line_length=5, line_gap=3)

    # Righr side or left side
    b_size = 5
    n_row, n_col = image.shape
    side = 0
    if np.sum(image[0:b_size,0:b_size]) < np.sum(image[0:b_size,n_col-b_size:n_col]):
        side = 1

    # triming lines
    t_lines = line_triming(lines,side,n_row, n_col)

    # create muscle mask
    if len(t_lines)>0:
        mask = mmask(image.shape, t_lines[0], side)
        image[mask] = 0

    # plot
    if visulization == True:

        plt.figure(figsize=(8, 4))

        plt.subplot(141)
        plt.imshow(image, cmap=plt.cm.gray)
        plt.title('Input image')

        plt.subplot(142)
        plt.imshow(mask, cmap=plt.cm.gray)
        plt.title('smoothing image')

        plt.subplot(143)
        plt.imshow(edges, cmap=plt.cm.gray)
        plt.title('Canny edges')

        plt.subplot(144)
        plt.imshow(edges * 0)
      
        for line in t_lines:
            p0, p1 = line
            #print (line,np.double(p0[0] - p1[0])/np.double(p0[1]- p1[1]))
            plt.plot((p0[0], p1[0]), (p0[1], p1[1]))

        plt.title('Probabilistic Hough')
        plt.axis('image')
        plt.show()
   
    return image


def test_example():
        
    # Construct toydata
    image = np.zeros((100, 100))
    idx = np.arange(25, 75)
    image[idx[::-1], idx] = 255
    image[idx, idx] = 255
    
    # Classic straight-line Hough transform  
    h, theta, d = hough_line(image)

    # plot
    plt.figure(figsize=(8, 4))

    plt.subplot(131)
    plt.imshow(image, cmap=plt.cm.gray)
    plt.title('Input image')

    plt.subplot(132)
    plt.imshow(np.log(1 + h),
               extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]),
                       d[-1], d[0]],
               cmap=plt.cm.gray, aspect=1/1.5)
    plt.title('Hough transform')
    plt.xlabel('Angles (degrees)')
    plt.ylabel('Distance (pixels)')

    plt.subplot(133)
    plt.imshow(image, cmap=plt.cm.gray)
    rows, cols = image.shape
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
        y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
        y1 = (dist - cols * np.cos(angle)) / np.sin(angle)
        plt.plot((0, cols), (y0, y1), '-r')
    plt.axis((0, cols, rows, 0))
    plt.title('Detected lines')

    # camera man, using the Probabilistic Hough Transform
    image = data.camera()
    edges = canny(image, 2, 1, 25)
    lines = probabilistic_hough_line(edges, threshold=10, line_length=5, line_gap=3)

    plt.figure(figsize=(8, 3))

    plt.subplot(131)
    plt.imshow(image, cmap=plt.cm.gray)
    plt.title('Input image')

    plt.subplot(132)
    plt.imshow(edges, cmap=plt.cm.gray)
    plt.title('Canny edges')

    plt.subplot(133)
    plt.imshow(edges * 0)

    for line in lines:
        p0, p1 = line
        plt.plot((p0[0], p1[0]), (p0[1], p1[1]))

    plt.title('Probabilistic Hough')
    plt.axis('image')
    plt.show()
