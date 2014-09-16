import scipy
import numpy as np
import cv2
import cv
#import matplotlib.pyplot as plt
import math
#import Saveimage
#import QgsGeometry
from numpy import ones,vstack
from numpy.linalg import lstsq
import Line
from scipy import ndimage
from scipy import misc
import PIL
from PIL import Image, ImageDraw
import sys

#import shapely
#from shapely.geometry import LineString


print scipy.version.version
print (sys.path)

print np.version.version
print scipy.version.version


def edgeenumeration(imdata,segmentdata):
    segmentdata=segmentdata.astype(np.uint8)
    ret,thresh=cv2.threshold(segmentdata,127,255,0)
    image, contours, hirerarchy =cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    L=list(enumerate(contours))
    return L
def normalization(p1,p2 ):
    
    dx=p2[0]-p1[0]
    
    dy=p2[1]-p1[1]
    Pre1=(-dy,dx)
    Pre2=(dy,-dx)
    #line=LineString(((-dy,dx),(dy,-dx)))
    points=[Pre1,Pre2]
    #line = Line(data)
    x_coords, y_coords = zip(*points)
    A = vstack([x_coords,ones(len(x_coords))]).T
    m, c = lstsq(A, y_coords)[0]
    #print "Line Solution is y = {m}x + {c}".format(m=m,c=c)
    
    
    return  m,c
averg=[] 
    #L=edgeenumeration(imdata,segmentdata)
    #k=12
    #length=len(L)
    #for i  in range(12,length-12):
    #'first pixel'
     #p1=L[i-12]  
    #'second pixel'         
     #p2=L[i+12]   
     #plt.plot(p1,p2)  
     
     
RBSTimage=np.zeros(shape=(2304,1000)) 
   
def interpolation(imdata,segmentdata ):
     segmentdata=segmentdata.astype(np.uint8)
     #segmentdata = cv2.cvtColor(segmentdata, cv2.COLOR_BGR2GRAY)
     ret,thresh=cv2.threshold(segmentdata,127,255,0)
     contours, hirerarchy =cv2.findContours(thresh, 1, 2)
     
     #imdata = cv2.drawContours(imdata, [cnt], 0, (0,255,0), 3)
    
     
     
     for i in range(0,len(contours)-1):
        cnt=contours[i]
        #print cnt
        averg.append(cnt[0,0])
         
         #Size=cnt.shape
         #print Size[0],Size[1],Size[2]
         
         #print cnt[0,0][1],cnt[1,0][0], cnt[2]
        # averg.append(imdata[cnt[0,0][1],cnt[0,0][0]+1]+imdata[cnt[0,0][1]+1,cnt[0,0][0]+1]/2)
         

         
     L=list(enumerate(averg))
     im=[]       
     length=len(L)
     for i  in range(12,length-12):
         #plt.hold(True)
         p1=L[i-12]
         #print p1
         pn1=p1[1]
           
         'second pixel'         
         p2=L[i+12] 
         #pn2=np.asarray(p2)
         pn2=p2[1]
         
         m,c=normalization(pn1,pn2)
         p=L[i]
         p=p[1]
         x1=p[0]
         y1=p[1]
         for j in range(100):
             
             y=m*j+c
             #x2=Ponpre[0]
             #y2=Ponpre[1]
             dist = math.sqrt( (j - x1)**2 + (y - y1)**2 )
             dist=math.floor(dist)
             size=imdata.shape
             #print size
             #print i
             if (dist<size[0]-2 and i<size[1]-1):
                 
                 RBSTimage[dist][i]=(imdata[dist+1,i-1]+imdata[dist+2,i-1])/2
                 
               #cnt=imdata[dist,i]
               
               #(imdata[dist,i+1]+imdata[dist+1,i])/2
               #RBSTimage[i][j]=(imdata[cnt[0,0][1],cnt[0,0][0]+1]+imdata[cnt[0,0][1]+1,cnt[0,0][0]+1]/2)
               
               
                 
                          
         #print Size[0],Size[1],Size[2]
         
         #print cnt[0,0][1],cnt[1,0][0], cnt[2]
           
          
         
         #plt.plot(p1,p2)
     #Saveimage.save("RBST", ext="tif", close=False, verbose=True)    
     #RBSTimage=plt.show()
     #RBSTimage= misc.imsave('fileName.tif', RBSTimage)
     #NEW_X_SIZE=len(RBSTimage)
     #NEW_Y_SIZE=len(RBSTimage)
     #new_img = Image.new("L", (NEW_X_SIZE, NEW_Y_SIZE), "white")
     #new_img.putdata(RBSTimage)
     #new_img.save('out.tif')
     #RBSTimage=im.save('RBSTimage.tiff')
        #im=plt.plot(p1,p2)
         #image=plotAxes(im, 180, p1, p2)
        #arrim=np.asarray(im)
        #print arrim.shape
     #RBSTimage= misc.imsave('RBST.tif', im)
# '''after calculating RBST we use sobel filtering'''
     return RBSTimage     
         
def sobelfiltering(imdata,segmentdata):
     RBSTimage=interpolation(imdata,segmentdata )
     RBSTimage=np.array(RBSTimage)
     print RBSTimage.shape
     #RBSTimage=([[ 0.,  0.,  0.,  0.,  0.],
       #[ 0.,  0.,  0.,  0.,  0.],
       #[ 0.,  0.,  0.,  0.,  0.],
       #[ 0.,  0.,  0.,  0.,  0.],
       #[ 0.,  0.,  0.,  0.,  0.]])#
     RBSTimage=np.array(RBSTimage)
     im = RBSTimage.astype('int32')
     dx = ndimage.sobel(im, 0)  # horizontal derivative
     dy = ndimage.sobel(im, 1)  # vertical derivative
     mag = np.hypot(dx, dy)  # magnitude
     
     mag *= 255.0 / np.max(mag)  # normalize (Q&D)
     #misc.imsave('sobel.tff', mag)
 #    Saveimage.save("sobel", ext="tif", close=False, verbose=True) 
     
         
     return mag
def shortrunfacility(imdata, segmentdata):
     soblresult=sobelfiltering(imdata,segmentdata)
     '''we want to define short run and long run during horizontal direction'''
     Size=soblresult.shape
     width=Size[1]
     print width
     
     heigth=Size[0]
     print heigth
     counter=0
     runmatrix=np.zeros(Size)
     print RBSTimage
     for i in range (0, heigth):
         counter=0
         for j in range (0, heigth-1):
             for k in range (0, width-1):
               for l in range (0, width-1):
                  if soblresult[k,l]==i:
                     counter=counter+1
             runmatrix[i,j]=counter
             print runmatrix.shape
             return runmatrix              
         
                     
                     
                     
     
     
            
        
        

    
    
    
