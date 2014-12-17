
"""The Tomosynthesis Image Class"""


import numpy as np
class TImage:
 
    def __init__ (self, im = None):
        """Initialization function"""

        self.dim = 0
        self.size_0 = 0
        self.size_1 = 0
        self.size_2 = 1
        self.data_type = None
        self.data = []
        self.sampled_data = []
        
        self.patchesList = []
        self.feats = []

        self.predicts = []

    def __repr__(self):
        """__repr__ function"""
        return 'TImage(%r, %r, %r)' % (self.size_0, self.size_1, self.size_2)

    def setDim(self,dim):
        """Set dimensionality of the image

        Parameters
        ----------
        dim : integer
            The dimensionality.
        
        """
        self.dim = dim

    def setSize(self, size_0, size_1, size_2):
        """Set size of the image

        Parameters
        ----------
        size_0 : integer
            The number of rows.
        size_1 : integer
            The number of columns.
        size_2 : integer
            The number of slices.
        
        """
        self.size_0 = size_0
        self.size_1 = size_1
        self.size_2 = size_2

        self.patchesList = [None] * size_2
        self.feats = [None] * size_2

        self.predicts = [None] * size_2

    def setDataType(self,data_type):
        """Set datatype of the image

        Parameters
        ----------
        data_type : integer
            The data type of the image.
        """
        self.data_type = data_type

    def setData(self,data):
        """Set data of the image

        Parameters
        ----------
        data : numppy array
            The data of the image.
        """
        self.data = data
        
    def getDim(self):
        """ Get the dimensionality of the image"""
        return self.dim

    def getSize(self):
        """ Get the size of the image"""
        return (self.size_0, self.size_1, self.size_2)

    def getDataType(self):
        """ Get the data type of the image"""
        return self.data_type

    def getData(self):
        """ Get the data of the image"""
        return self.data

    def downSample(self, rate):
        """Down sampling to the image

        Parameters
        ----------
        rate : interget
            The sample rate.
        """

        rows = np.array(range(0,self.size_0,rate))
        cols = np.array(range(0,self.size_1,rate))
        
        for i in range(self.size_2):
            rs = self.data[i][rows,:]
            cs = rs[:,cols]
            self.sampled_data.append(cs)

            
class TImageSlice:
    """ The light TImage class"""
    
    def __init__ (self):
        
        self.LightPatchList = []
        self.predicts = None
        self.feats = None

    def __repr__(self):
        return 'TImageSlice(%r)' % (len(self.LightPatchList))
