import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import misc
from scipy import ndimage

def canny(imdata):
    
    misc.imsave('fileName.tif', imdata)
    image =ndimage.imread('fileName.tif',0)
    edges = cv2.Canny(image,1,100)
    
    
    #edges = cv2.Canny(imdata,1,100)
    plt.subplot(121),plt.imshow(imdata,cmap = 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

    plt.show()

#img = cv2.imread('messi5.jpg',0)
    

