"""The class for image patch"""

import numpy as np
from scipy import sqrt, pi, arctan2, cos, sin
import matplotlib.pyplot as plt
import tiffLib

class TPatch:

    pdata = None
    image_center = None
    patch_center = None
    data_size = None
    patch_size = None
      
    rings = []
    numrings = None
    m_features = None
    v_features = None

    gx = None
    gy = None
    gmagnitude = None
    gorientation = None
    gnormorientation = None
    greflorientation = None
    location_ori = None

    numsectors = None
    gsectors = []
    normal_percentage = None
    normal_level = None
    
    

    def __init__ (self):
        '''Initialization'''

    def initialize(self,imdata):
        '''Initialization with imagedata'''
        self.pdata = imdata
        self.data_size = imdata.shape
        
        self.patch_center = (imdata.shape[0]/2, imdata.shape[1]/2)
        self.patch_size = max(imdata.shape[0],imdata.shape[1])/2
        

    def getRing(self, radius1, radius2):

        nr,nc = self.data_size
        X, Y = np.ogrid[0:nr-1, 0:nc-1]

        mask1 = (X - self.patch_center[0])**2 + (Y - self.patch_center[1])**2 < radius1**2
        mask2 = (X - self.patch_center[0])**2 + (Y - self.patch_center[1])**2 < radius2**2

        mask = mask2-mask1
        ring = self.pdata[mask]
        
        return ring

    def getRings(self,numrings):

        self.rings = []
        r1 = 0
        self.numrings = numrings
        dr = self.patch_size/numrings
        for i in range(numrings):
            r2 = r1 + dr
            ring = self.getRing(r1,r2)
            self.rings.append(ring)
            r1 = r1 + dr
            dr = dr - 1


    def getRingsMeanFeats(self):

        self.m_features = np.zeros((1,self.numrings), dtype=np.double)
        for i in range(self.numrings):
            self.m_features[0,i] = np.mean(self.rings[i])

    def getRingsVarFeats(self):

        self.v_features = np.zeros((1,self.numrings), dtype=np.double)
        for i in range(self.numrings):
            self.v_features[0,i] = np.std(self.rings[i])

    def dumpRingsFeats(self):

        feat_table = np.hstack((self.m_features,self.v_features))

        return feat_table

    def computeGradient(self):
        
        self.gx = np.zeros(self.data_size,dtype=np.double)
        self.gy = np.zeros(self.data_size,dtype=np.double)
        self.gx[:, :-1] = np.diff(np.double(self.pdata), n=1, axis=1)
        self.gy[:-1, :] = np.diff(np.double(self.pdata), n=1, axis=0)

        self.gmagnitude = sqrt(self.gx**2 + self.gy**2)
        self.gorientation = arctan2(self.gy, self.gx) * (180 / pi) % 180
        

    def gradOrieNormalize(self, threshold = 1200):

        nr,nc = self.data_size
        self.gnormorientation = np.zeros(self.data_size,dtype=np.double)
        self.location_ori = np.zeros(self.data_size,dtype=np.double)
        self.greflorientation = np.zeros(self.data_size,dtype=np.double)

        print self.data_size

        for i in range(nr):
            y = i - self.patch_center[0]
            for j in range(nc):
                x = j - self.patch_center[1]
                self.location_ori[i][j] = arctan2(abs(y), abs(x)) * (180 / pi) % 180
                
                # first & third
                if x*y <=0:
                    self.greflorientation[i][j] = self.gorientation[i][j]
                    
                # second & forth
                else:
                    self.greflorientation[i][j] = 180 - self.gorientation[i][j]
                    
        self.gnormorientation = 180 - abs(self.greflorientation - self.location_ori)
        mask = self.gmagnitude < threshold
        self.gnormorientation[mask] = 0
        

    def getSectorMask(self, angle_range, radius = 'default'):

        if radius == 'default':
            radius = self.patch_size

        shape = self.data_size
        centre = self.patch_center
        x,y = np.ogrid[:shape[0],:shape[1]]
        cx,cy = centre
        tmin,tmax = np.deg2rad(angle_range)

        # ensure stop angle > start angle
        if tmax < tmin:
                tmax += 2*np.pi

        # convert cartesian --> polar coordinates
        r2 = (x-cx)*(x-cx) + (y-cy)*(y-cy)
        theta = np.arctan2(x-cx,y-cy) - tmin

        # wrap angles between 0 and 2*pi
        theta %= (2*np.pi)

        # circular mask
        circmask = r2 <= radius*radius

        # angular mask
        anglemask = theta <= (tmax-tmin)

        return circmask*anglemask

    def getGSectors(self, numsectors):
     
        self.numsectors = numsectors
        dangle = 360/numsectors

        self.gsectors = []

        angle1 = 0
        for i in range(numsectors):
            angle2 = angle1 + dangle
            mask = self.getSectorMask((angle1,angle2))
            sector = self.gnormorientation[mask]
            angle1 = angle1 + dangle

            self.gsectors.append(sector)

    def getNormPerc(self, norm_th = 135):

        self.normal_percentage = np.zeros((1,self.numsectors), dtype=np.double)
        for i in range(self.numsectors):
            mask = self.gsectors[i] > norm_th
            norm_count = self.gsectors[i][mask].shape[0]
            total_count = self.gsectors[i].shape[0]
            self.normal_percentage[0,i] = np.double(norm_count)/np.double(total_count)

    def getNormLevl(self):

        self.normal_level = np.zeros((1,self.numsectors), dtype=np.double)
        for i in range(self.numsectors):
            
            norm_total = np.sum(self.gsectors[i])
            mask = self.gsectors[i] > 0
            total_count = self.gsectors[i].shape[0]
            self.normal_level[0,i] = np.double(norm_total)/np.double(total_count)

       
        
                
        
        
        
            
