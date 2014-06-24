#
# The Tomosynthesis Image Class
#

class TImage:

    dim = 0
    size_0 = 0
    size_1 = 0
    size_2 = 0
    data_type = None
    data = []
 
    def __init__ (self, im = None):
        '''Initialization'''
        self = im

    def setDim(self,dim):
        self.dim = dim

    def setSize(self, size_0, size_1, size_2):       
        self.size_0 = size_0
        self.size_1 = size_1
        self.size_2 = size_2

    def setDataType(self,data_type):
        self.data_type = data_type

    def setData(self,data):
        self.data = data
        
    def getDim(self):
        return self.dim

    def getSize(self):
        return (self.size_0, self.size_1, self.size_2)

    def getDataType(self):
        return self.data_type

    def getData(self):
        return self.data









