"""  The class for micro calcification """

import numpy as np
import tiffLib

class TMicroCal:

    label = None
    center = []
    intensity = None
    area = None
    roi = None
    global_id = None
    global_flag = None

    neighbours_2d = []
    neighbour_dis_2d = []
    density_2d = None
    num_neighbours_2d = None

    def __init__ (self):
        self.center = np.zeros((3,), dtype=np.int)
        self.global_flag = False
        '''Initialization'''

    def computeDensity_2d(self):
        
        self.num_neighbours_2d = len(self.neighbours_2d)
        dis_sum = sum(self.neighbour_dis_2d)
        self.density_2d = dis_sum/self.num_neighbours_2d

class TMicroCal_3D:

    global_id = None
    center = None
    intensity = None
    volume = None

    density = None
    num_neighbours = None

    def __init__ (self):
        '''Initialization'''
        

    

    
