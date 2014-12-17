"""Thin plate spline interpolation."""

import numpy as np
import math
import cv2
import time


class TPSpline:

    def __init__ (self):

        self.psrc = []
        self.pdst = []
        
        self.lenspline = None     
        self.mapx = None
        self.mapy = None
        self.cMatrix = None

        self.mx = None
        self.my = None

    def __repr__(self):
        return 'TPSpline'

    def setCorrespondences(self, pS, pD):
        """Set the fiducial points for registration.

        Parameters
        ----------
        pS : list
            Lists of control point from source image
        pD : list
            Lists of control point from destination image

        """

        if len(pS) != len(pD):
            print 'Correspondences not consistent !'

        self.psrc = pS
        self.pdst = pD

        self.lenspline = min(len(pS),len(pD))

    def fktU(self, p1, p2):
        """The U = f(r) function"""

        r = (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

        if r == 0:
            return 0.0
        
        else:
            return r * math.log(r)
                 

    def computeSplineCoeffs(self, lamda):
        """Solve the linear system"""

        dim = 2
        n = self.lenspline

        # initialize matrices
        V = np.zeros((dim,n+dim+1),np.double)
        P = np.ones((n,dim+1),np.double)
        K = np.eye(n, M=None, k=0, dtype=float)*lamda
        L = np.zeros((n+dim+1,n+dim+1),np.double);
        
        # fill up K
        for i in range(self.lenspline):
            for j in range(self.lenspline):
                if i != j:
                    K[i, j] = self.fktU(self.psrc[i], self.psrc[j])

        # fill up P
        P[:,1] = np.asarray(self.psrc)[:,0]
        P[:,2] = np.asarray(self.psrc)[:,1]

        # fill up L
        L[0:n, 0:n] = K
        L[0:n, n:n+dim+1] = P
        L[n:n+dim+1, 0:n] = np.transpose(P)

        # fill up V
        V[0,0:n] = np.asarray(self.pdst)[:,0]
        V[1,0:n] = np.asarray(self.pdst)[:,1]

        
        invL = np.linalg.inv(L)
        self.cMatrix = np.dot(invL,np.transpose(V))
        print self.cMatrix
                          

    def interpolate(self,p):
        """Compute displacement based on computed splie coefficients"""
       
        k1 = self.cMatrix.shape[0] - 3
        kx = self.cMatrix.shape[0] - 2
        ky = self.cMatrix.shape[0] - 1

        a1, ax, ay, cTmp, uTmp, tmp_i, tmp_ii = (0, 0, 0, 0, 0, 0, 0)

        y, x = (0,0)
        for i in range(2):
            a1 = self.cMatrix[k1,i]
            ax = self.cMatrix[kx,i]
            ay = self.cMatrix[ky,i]

            tmp_i = a1 + ax * p[0] + ay * p[1]
            tmp_ii = 0

            for j in range(self.lenspline):
                cTmp = self.cMatrix[j,i]
                uTmp = self.fktU(self.psrc[j], p)
                tmp_ii = tmp_ii + (cTmp * uTmp)

            if i == 0:
                y = tmp_i + tmp_ii
            if i == 1:
                x = tmp_i + tmp_ii
                
        interP = (y,x)
        #print interP
        return interP

    def interpolate_fast(self,p):
        """A faster version of Compute displacement based on computed
        splie coefficients"""

        psudo_p = np.asarray([1,p[0],p[1]])

        dim = self.cMatrix.shape[0]        
        aff_coeff_x = self.cMatrix[(dim - 3):dim,0]
        aff_coeff_y = self.cMatrix[(dim - 3):dim,1]

        distances = [self.fktU(self.psrc[j], p) for j in xrange(self.lenspline)]
        distances = np.asarray(distances)

        tmp_i = np.sum(aff_coeff_x*psudo_p)
        tmp_ii = np.sum(self.cMatrix[0:self.lenspline,0]*distances)
        y = tmp_i + tmp_ii

        tmp_i = np.sum(aff_coeff_y*psudo_p)
        tmp_ii = np.sum(self.cMatrix[0:self.lenspline,1]*distances)
        x = tmp_i + tmp_ii

        intP =(y,x)
        #print intP
        return intP 

    def warpImage(self, src, lamda = 0.05):
        """Warpiing the source image"""

        self.computeSplineCoeffs(lamda)

        print 'done compute spline coeeficients'

        start = time.clock()
        print start
        self.computeMaps(src.shape)
        elapsed = (time.clock() - start)
        print str(elapsed) + 's'

        print 'done compute map'

        warped = cv2.remap(src, self.my, self.mx, cv2.INTER_CUBIC)

        print 'done warping'

        return warped
 
    def computeMaps(self, datasize):
        """Compute dispalcement for all pixels"""

        self.mx = np.zeros(datasize, np.float32)
        self.my = np.zeros(datasize, np.float32)

        for row in range(datasize[0]):
            for col in range(datasize[1]):
                intP = self.interpolate((col,row))              
                self.mx[row, col] = intP[1]
                self.my[row, col] = intP[0]


    def computeMaps_fast(self, datasize):
        """A faster version of compute dispalcement for all pixels"""

        intPs = [self.interpolate_fast((col,row))for row in xrange(datasize[0]) for col in xrange(datasize[1])]             
        intPs = np.asarray(intPs)
        
        self.mx = np.float32(intPs[:,1].reshape(datasize))
        self.my = np.float32(intPs[:,0].reshape(datasize))
        
                



















        
