import numpy as np
import tiffLib

class TMicroCal:
    """ The class for micro calcification in slice.

    Variables
    ----------
    label: interger
        The id of the clacification in the current slice
    intensity: float
        The mean intensity value of the clacification
    area: integer
        The area value of the clacification
    global_id: integer
        The global id of the clacification in the volume
    neighbours_2d: list of intergers
        The list of its neighbours in the current slice
    neighbour_dis_2d: list of floats
        The list of distances to its neighbours in the current slice
    density_2d: float
        The calcification density in the neioghbourhood.

    """

    def __init__ (self):

        self.label = None
        self.intensity = None
        self.area = None
        self.roi = None
        self.global_id = None

        self.neighbours_2d = []
        self.neighbour_dis_2d = []
        self.density_2d = None
        self.num_neighbours_2d = None
        
        self.center = np.zeros((3,), dtype=np.int)
        self.global_flag = False

    def __repr__(self):
        return 'TMicroCal(%r, %r)' % (self.global_id, self.area)

    def computeDensity_2d(self):
        """ Compute the calcification density of the neighbourhoodd"""
        self.num_neighbours_2d = len(self.neighbours_2d)
        dis_sum = sum(self.neighbour_dis_2d)
        self.density_2d = dis_sum/self.num_neighbours_2d

class TMicroCal_3D:
    """ The class for micro calcification in volume.

    Variables
    ----------
    global_id: integer
        The global id of the clacification in the volume
    center: a tuple of three integers
        The (x,y,x) coordination of the calcification in the volume
    intensity: float
        The mean intensity value of the calcification blob
    volume: integer
        The volume of the calcification blob
    density: float
        The density of the neighbouthood (3D)
    num_neighbours: integer
        The number of the neighbours in the volume
 
    """

    def __init__ (self):

        self.global_id = None
        self.center = None
        self.intensity = None
        self.volume = None

        self.density = None
        self.num_neighbours = None

    def __repr__(self):
        return 'TMicroCal_3D(%r, %r)' % (self.global_id, self.volume)
        
        

    

    
