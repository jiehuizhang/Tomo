
import numpy as np
import skimage
import scipy
import mahotas as mh
import cv2
import matplotlib.pyplot as plt

def circulaity(imagedata):
 #image=cv2.imread('',0)

 plt.plot(imagedata)
 newimage=imagedata.astype(np.uint8)
 ret,thresh=cv2.threshold(newimage, 127, 255, 0)
 
 contours,hiearchy=cv2.findContours(thresh, 1, 2)
 
 cnt=contours[0]
 area=cv2.contourArea(cnt)
 print area
 perimeter=cv2.arcLength(cnt,True)
 return perimeter*perimeter/area
 
 
    
