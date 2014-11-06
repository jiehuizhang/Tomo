"""  The class for micro calcification """

import numpy as np
import tiffLib

class TMicroCal:

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
        
        self.num_neighbours_2d = len(self.neighbours_2d)
        dis_sum = sum(self.neighbour_dis_2d)
        self.density_2d = dis_sum/self.num_neighbours_2d

class TMicroCal_3D:

    def __init__ (self):

        self.global_id = None
        self.center = None
        self.intensity = None
        self.volume = None

        self.density = None
        self.num_neighbours = None

    def __repr__(self):
        return 'TMicroCal_3D(%r, %r)' % (self.global_id, self.volume)
        
        

    

    
