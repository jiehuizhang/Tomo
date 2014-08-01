"""The class for image patch"""

import numpy as np
from numpy import linalg as LA
from scipy import sqrt, pi, arctan2, cos, sin
from scipy import stats
import matplotlib.pyplot as plt
from sklearn import linear_model
import tiffLib
import math

class TPatch:

    pdata = None
    image_center = None
    patch_center = None
    data_size = None   # actual size
    patch_size = None
      
    rings = []
    numrings = None
    m_features = None
    v_features = None
    ringLinearSlopeMean = None
    ringLinearSlopeVar = None

    gx = None
    gy = None
    gmagnitude = None
    gorientation = None
    gnormorientation = None
    greflorientation = None
    location_ori = None
    nor_perc_max = None
    nor_perc_mean = None
    nor_perc_std = None
    nor_lev_mean = None
    nor_lev_max = None
    nor_lev_std = None
    norGra_level_mean = None
    norGra_level_max = None
    norGra_level_std = None
    norGra_level = None

    numsectors = None
    gsectors = []
    normal_percentage = None
    normal_level = None
    
    PSD_polar = None
    PSD = None

    def __init__ (self):
        '''Initialization'''
        self.pdata = None
        self.image_center = None
        self.patch_center = None
        self.data_size = None   # actual size
        self.patch_size = None
      
        self.rings = []
        self.numrings = None
        self.m_features = None
        self.v_features = None
        self.ringLinearSlopeMean = None
        self.ringLinearSlopeVar = None

        self.gx = None
        self.gy = None
        self.gmagnitude = None
        self.gorientation = None
        self.gnormorientation = None
        self.greflorientation = None
        self.location_ori = None
        self.nor_perc_max = None
        self.nor_perc_mean = None
        self.nor_lev_mean = None
        self.nor_lev_max = None
        self.norGra_level_mean = None
        self.norGra_level_max = None
        self.norGra_level_std = None
        self.norGra_level = None

        self.numsectors = None
        self.gsectors = []
        self.normal_percentage = None
        self.normal_level = None
        
        self.PSD_polar = None
        self.PSD = None

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
            dr = max(dr - 2,5)


    def getRingsMeanFeats(self):

        self.m_features = np.zeros((1,self.numrings), dtype=np.double)
        for i in range(self.numrings):
            self.m_features[0,i] = np.mean(self.rings[i])

    def LinearRegRingMeanFeats(self):
        
        sequence = np.arange(self.m_features.shape[1])     
        slope, intercept, r_value, p_value, std_err = stats.linregress(sequence,self.m_features[0])
        self.ringLinearSlopeMean = slope
        return slope

    def getRingsVarFeats(self):

        self.v_features = np.zeros((1,self.numrings), dtype=np.double)
        for i in range(self.numrings):
            self.v_features[0,i] = np.std(self.rings[i])

    def LinearRegRingVarFeats(self):
        
        sequence = np.arange(self.v_features.shape[1])     
        slope, intercept, r_value, p_value, std_err = stats.linregress(sequence,self.v_features[0])
        self.ringLinearSlopeVar = slope
        return slope

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

        self.gmagnitude[mask] = 0
        

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
            #sector = self.gnormorientation[mask]
            # test using magnitude
            sector = self.gmagnitude[mask] 
            
            angle1 = angle1 + dangle

            self.gsectors.append(sector)

    def getNormPerc(self, norm_th = 135):
        
        self.normal_percentage = np.zeros((1,self.numsectors), dtype=np.double)
        for i in range(self.numsectors):
            mask = self.gsectors[i] > norm_th
            norm_count = self.gsectors[i][mask].shape[0]
            total_count = self.gsectors[i].shape[0]
            self.normal_percentage[0,i] = np.double(norm_count)/np.double(total_count)
        self.nor_perc_mean = np.mean(self.normal_percentage)
        self.nor_perc_max = np.max(self.normal_percentage)
        self.nor_perc_std = np.std(self.normal_percentage)
        

    def getNormLevl(self):

        self.normal_level = np.zeros((1,self.numsectors), dtype=np.double)
        for i in range(self.numsectors):
            
            norm_total = np.sum(self.gsectors[i])
            mask = self.gsectors[i] > 0
            total_count = self.gsectors[i].shape[0]
            self.normal_level[0,i] = np.double(norm_total)/np.double(total_count)
        self.nor_lev_mean = np.mean(self.normal_level)
        self.nor_lev_max = np.max(self.normal_level)
        self.nor_lev_std = np.std(self.normal_level)


    def getNormGradmagnitude(self):

        self.norGra_level = np.zeros((1,self.numsectors), dtype=np.double)
        for i in range(self.numsectors):
            
            norm_total = np.sum(self.gsectors[i])
            self.norGra_level[0,i] = np.double(norm_total)
        self.norGra_level_mean = np.mean(self.norGra_level)
        self.norGra_level_max = np.max(self.norGra_level)
        self.norGra_level_std = np.std(self.norGra_level)

    def padding(self,padded_size):

        if self.data_size[0] >= padded_size:
            vertical = 0
            if self.data_size[1] >= padded_size:
                horizontal = 0              
            else:
                horizontal = padded_size - self.data_size[1]
        if self.data_size[1] >= padded_size:
            horizontal = 0
            if self.data_size[0] >= padded_size:
                vertical = 0
            else:
                vertical = padded_size - self.data_size[0]
        if self.data_size[0] < padded_size and self.data_size[1] < padded_size:         
            vertical = padded_size - self.data_size[0]
            horizontal = padded_size - self.data_size[1]
        
        up = vertical/2
        down = vertical - up
        
        right = horizontal/2
        left = horizontal - right

        return np.lib.pad(self.pdata, ((up,down),(left,right)),'edge')

    def converCARToPOL(self,PSD, freq = 128):

        numr, numc = PSD.shape
        
        # mapping from cartisaian to polar coordinate 
        polarPSD = np.zeros((freq,180), dtype=np.double)
        counter = np.ones((freq,180), dtype=np.int)
        center = (PSD.shape[0]/2, PSD.shape[1]/2)
        for i in range(numr/2):
            for j in range(numc):
                theta = int(arctan2(center[0]-i, j-center[1]) * (180 / pi) % 180)
                r = int(sqrt((i - center[0])**2 + (j - center[1])**2))
                if r < freq:
                    polarPSD[r,theta] = polarPSD[r,theta] + PSD[i,j]
                    counter[r,theta] = counter[r,theta] + 1
        polarPSD = polarPSD/counter

        # filling zero entries
        for i in range(polarPSD.shape[0]):
            vector = polarPSD[i,:]
            newvector = np.zeros(vector.shape)
            validid = vector>0
            values = vector[validid]
            if values.shape[0] == 0:
                values = np.ones((1,),np.double)*0.001

            # creat shifted ids
            tempmat = np.zeros(validid.shape)
            applied_id = tempmat>0
            previous = 0
            current = 0
            for t in range(validid.shape[0]):
                if validid[t] == True:
                    current = t
                    if previous > 0:
                        applied_id[(current + previous)/2]=True
                    previous = current
            # filling          
            counter = 0
            for j in range(vector.shape[0]):
                val = values[counter]
                if validid[j] == True:
                    counter = min(counter + 1,values.shape[0]-1)             
                newvector[j] = val
                
            polarPSD[i,:] = newvector

        return polarPSD    
  
    def getFD(self, padded_size = 321, fqrange = (4,90)):

        # zero padding
        padded_data = self.padding(padded_size)

        # compute PSD
        shiftedFT = np.fft.fftshift(np.fft.fft2(padded_data))
        self.PSD = np.abs(shiftedFT**2)

        # convert to polar space
        self.PSD_polar = self.converCARToPOL(self.PSD,(padded_size - 1) /2)

        # transform to 1-d frequency function
        s_f = np.sum(self.PSD_polar,1)/180

        # linear regression
        valid_sf = s_f[fqrange[0]:fqrange[1]]
        freqband = np.arange(fqrange[0],fqrange[1])
        freqband = 1.0/np.double(freqband)      
        slope, intercept, r_value, p_value, std_err = stats.linregress(np.log10(freqband),np.log10(valid_sf))

        FD = (8 - slope)/2

        # transform to 1-d angular function
        valid_polarspd = self.PSD_polar[fqrange[0]:fqrange[1],:]
        s_theta = np.sum(self.PSD_polar,0)/valid_polarspd.shape[0]
        norm_s_theta = s_theta/LA.norm(s_theta,1)

        # entropy of angular spread power
        H = np.sum(-norm_s_theta * np.log2(norm_s_theta))

        # varaince of angular spread power
        mean = np.mean(norm_s_theta)
        varaince = np.sum((norm_s_theta - mean)**2)/norm_s_theta.shape[0]

        # std of angular spread power
        theta = np.arange(0,180)
        m_theta = np.sum(theta*norm_s_theta)
        std = sqrt(np.sum(norm_s_theta*((theta - m_theta)**2)))
        
        return (FD,H,varaince,std)
        
        


       
        
                
        
        
        
            
