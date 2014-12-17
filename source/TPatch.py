

import numpy as np
from numpy import linalg as LA
from scipy import sqrt, pi, arctan2, cos, sin
from scipy import stats
from scipy import fftpack
from sklearn import linear_model
import scipy.ndimage.filters as filters
import histEqualization
import tiffLib
import math

import histEqualization
import AT_denoising
import activeContourSegmentation as acSeg
import morphsnakes

class TPatch:
    """Image patch class. (A region of interest)

    Variables
    ----------
    pdata: 2D numpy array
        The patch data
    image_center: a tuple of two intergers
        The (x,y) coordinate in the oroginal image
    patch_center: a tuple of two intergers
        The (x,y) coordinate of the center of the patch
    data_size: a tuple of two intergers
        The shape of hte data array
    patch_size: integer
        The radius of the patch

    bagID: interger
        Used in the multi-instance learning classification.
        The ID of the training sample
    instanceID: interger
        Used in the multi-instance learning classification.
        The ID asigned by the classifier
   
    """

    def __init__ (self):
        """Initialization function"""
        self.pdata = None
        self.image_center = None
        self.patch_center = None
        self.data_size = None   # actual size
        self.patch_size = None

        self.downsampled = None
        self.gmagnitude = None
		
        self.bagID = None
        self.instanceID = None

    def __repr__(self):
        """__repr__ function"""
        return 'TPatch' 

    def initialize(self,imdata):
        """Initialization with imdata

        Parameters
        ----------
        imdata : numpy array
            Initilial data array.
        """
        self.pdata = imdata
        self.data_size = imdata.shape
        
        self.patch_center = (imdata.shape[0]/2, imdata.shape[1]/2)
        self.patch_size = max(imdata.shape[0],imdata.shape[1])/2

    def downSampling(self,rate = 2):
        """Down sampling the image patch by given sample rate

        Parameters
        ----------
        rate : integer
            The sample rate.
        """        
        rows = np.array(range(0,self.data_size[0],rate))
        cols = np.array(range(0,self.data_size[1],rate))

        rs = self.pdata[rows,:]
        self.downsampled = rs[:,cols]


    def getRings(self,data, numrings, dr_ini,r_ini = 0,delta_dr = 2):
        """Devide the image patch into a number of rings 

        Parameters
        ----------
        data : numpy array
            The patch data.
        numrings : integer
            The number of rings desired
        dr_ini : integer
            The initial delt value for radius increasment
        r_ini: integer
            The initial radious value
        delta_dr: integer
            The increase step of delt value  
        
        """        

        nr,nc = self.data_size
        X, Y = np.ogrid[0:nr-1, 0:nc-1]

        rings = []
        r1 = r_ini
        dr = self.patch_size/numrings
        dr = dr + dr_ini
        
        for i in range(numrings):
            r2 = r1 + dr      

            mask1 = (X - self.patch_center[0])**2 + (Y - self.patch_center[1])**2 < r1**2
            mask2 = (X - self.patch_center[0])**2 + (Y - self.patch_center[1])**2 < r2**2

            mask = mask2-mask1
            ring = data[mask]
            
            rings.append(ring)
            r1 = r1 + dr
            dr = max(dr - delta_dr,5)

            # for debugging
            '''
            outputPath = 'C:/Tomosynthesis/localtest/res/'
            temp = np.zeros(self.pdata.shape,np.double)
            temp[:,:] = data[:,:]
            temp[mask] = 0
            tiffLib.imsave(outputPath + str(i) + 'rings.tif',np.float32(temp))
            '''

        return rings

    def getSectors(self,data, numsects):
        """Devide the image patch into a number of angular sectors 

        Parameters
        ----------
        data : numpy array
            The patch data.
        numrings : integer
            The number of sectors desired
        
        """ 

        dangle = 360/numsects

        gsectors = []

        angle1 = 0
        for i in range(numsects):
            
            angle2 = angle1 + dangle
            
            radius = self.patch_size
            shape = self.pdata.shape

            x,y = np.ogrid[:shape[0],:shape[1]]
            cx,cy = self.patch_center
            tmin,tmax = np.deg2rad((angle1,angle2))

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

            mask = circmask*anglemask

            
            sector = data[mask]          
            angle1 = angle1 + dangle
            gsectors.append(sector)


        return gsectors

    def computeGradient(self, threshold = 1500):
        """Compute the gradient of the image patch 

        Parameters
        ----------
        threshold : integer
            Gradient magnitude below this value are set to be 0.
        
        """ 
        
        gx = np.zeros(self.data_size,dtype=np.double)
        gy = np.zeros(self.data_size,dtype=np.double)
        gx[:, :-1] = np.diff(np.double(self.pdata), n=1, axis=1)
        gy[:-1, :] = np.diff(np.double(self.pdata), n=1, axis=0)

        self.gmagnitude = sqrt(gx**2 + gy**2)
        mask = self.gmagnitude < threshold
        self.gmagnitude[mask] = 0

    def getIntenfeats(self, int_Rnum, int_Snum):
        """Compute intensity features of the image patch 

        Parameters
        ----------
        int_Rnum : integer
            The number of rings to be divided into.
        int_Rnum : integer
            The number of sectors to be divided into.
            
        """

        int_rings = self.getRings( self.pdata, int_Rnum, dr_ini = 5,r_ini = 15,delta_dr = 2)
        
        #### calculate mean
        mIntRing = np.zeros((1,int_Rnum), dtype=np.double)
        for i in range(int_Rnum):
            mIntRing[0,i] = np.mean(int_rings[i])
            
        # mean linear regration
        sequence = np.arange(mIntRing.shape[1])     
        slope, intercept, r_value, p_value, std_err = stats.linregress(sequence,mIntRing[0])
        intMLinear = slope

        # variance of mean
        var_mIntRing = np.std(mIntRing)

        #### calculate varriance
        vIntRing = np.zeros((1,int_Rnum), dtype=np.double)
        for i in range(int_Rnum):
            vIntRing[0,i] = np.std(int_rings[i])

        # variance linear regretion
        sequence = np.arange(vIntRing.shape[1])     
        slope, intercept, r_value, p_value, std_err = stats.linregress(sequence,vIntRing[0])
        intVLinear = slope

        '''intensity sector feats    '''
        int_sectors = self.getSectors(self.pdata, int_Snum)

        # calculate mean intensity power 
        mIntSec = np.zeros((1,int_Snum), dtype=np.double)
        for i in range(int_Snum):
            mIntSec[0,i] = np.mean(int_sectors[i])
        
        mIntSec = mIntSec/np.sum(mIntSec)
        intSsorted =  -np.sort(-mIntSec)

        threshold = 1.0/int_Snum
        mask = intSsorted > threshold
        int_power = intSsorted[mask].size

        angular_pdiff =  np.sum(intSsorted[:,0:4])-np.sum(intSsorted[:,8:int_Snum])

        #print int_power
        
        return  np.hstack((intMLinear, var_mIntRing, angular_pdiff, int_power))

    def getGradfeats(self, gr_Rnum, gr_Snum):
        """Compute gradient features of the image patch 

        Parameters
        ----------
        gr_Rnum : integer
            The number of rings to be divided into.
        gr_Snum : integer
            The number of sectors to be divided into.
            
        """

        ## gradient ring feats         ##################
        self.computeGradient(threshold = 2500)
        gr_rings = self.getRings( self.gmagnitude, gr_Rnum, dr_ini = 0,r_ini = 15,delta_dr = 1)

        # calculate mean
        mGrRing = np.zeros((1,gr_Rnum), dtype=np.double)
        for i in range(gr_Rnum):
            mGrRing[0,i] = np.mean(gr_rings[i])

        mGrRing = mGrRing/np.sum(mGrRing)
        maxGring = np.max(mGrRing[0:3])   #  this is a feature
        

        ## gradient sector feats         ##################
        gr_sectors = self.getSectors(self.gmagnitude, gr_Snum)

        # calculate mean
        mGrSect = np.zeros((1,gr_Snum), dtype=np.double)
        for i in range(gr_Snum):
            mGrSect[0,i] = np.mean(gr_sectors[i])

        mGrSect = mGrSect/np.sum(mGrSect)
        mGrSect = -np.sort(-mGrSect)

        angular_paccum = np.sum(mGrSect[:,0:6])  #  this is a feature
        angular_pdiff = np.sum(mGrSect[:,0:5])- np.sum(mGrSect[:,8:gr_Snum])  #  this is a feature

        threshold = 1.0/gr_Snum
        mask = mGrSect > threshold
        grSpower = mGrSect[mask].size    # this is a feature
      
        return np.hstack((maxGring, angular_paccum, angular_paccum, grSpower))

    def getSegmentFeats(self):
        """Compute gradient features of the image patch """

        ## preprocessing
        eqimg = histEqualization.histEqualization(self.pdata, 16)
        denoised = AT_denoising.DenoisingAW(eqimg)
        denoised = AT_denoising.DenoisingAW(denoised)
        imdata = AT_denoising.DenoisingAW(denoised)

        #segmentation ceter towards outside
        lsoutwards = acSeg.ac_outwards(imdata,visulization = False)

        # compute features based on segmentation
        outSegFeats = acSeg.getLabelImFeats(lsoutwards,center = (self.patch_center[0],self.patch_center[1]),orgim=imdata)
    
        return outSegFeats


class TLightPatch:
    """Light class of TPatch includes only class variables """

    def __init__ (self):
        self.pdata = None
        self.image_center = None
        self.patch_center = None
        self.data_size = None   # actual size
        self.patch_size = None
        self.feats = None
		
        self.bagID = None
        self.instanceID = None

    def __repr__(self):
        return 'TLightPatch' 


    


    
