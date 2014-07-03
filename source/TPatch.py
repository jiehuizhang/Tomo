"""The class for image patch"""

import numpy as np
import tiffLib

class TPatch:

    image_center = None
    patch_center = None
    data_size = None
    patch_size = None
    pdata = None
    rings = []
    features = []

    def __init__ (self):
        '''Initialization'''

    def getRing(self, radius1, radius2):

        nr,nc = self.data_size
        X, Y = np.ogrid[0:nr-1, 0:nc-1]

        mask1 = (X - self.patch_center[0])**2 + (Y - self.patch_center[1])**2 < radius1**2
        mask2 = (X - self.patch_center[0])**2 + (Y - self.patch_center[1])**2 < radius2**2

        mask = mask2-mask1
        ring = self.pdata[mask]

        ###for debugging
        self.pdata[mask] = 0
        path = 'C:/Tomosynthesis/localtest/res/'
        tiffLib.imsave(path + 'ring.tif',np.float32(self.pdata))
        ### done fo debugging
        
        return ring

    def getRings(self,numrings):

        rings = []

        r1 = 0
        dr = self.patch_size/numrings
        for i in range(numrings):
            r2 = r1 + dr
            rings.append(getRing,r1,r2)
            r1 = r1 + dr

        return rings
            
