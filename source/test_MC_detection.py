"""This script shows how to run micro calcification detection."""

import time
from multiprocessing import Pool
import gc

import numpy as np
import ImageIO
import TImage
import MC_Detection as mc
import tiffLib

def test_func():
    """Please specify the data directory and file name.This function runs
    the detection for a 3D stack.
    """

    dataPath = 'C:/Tomosynthesis/localtest/'
    outputPath = 'C:/Tomosynthesis/test_script/'   
    fileName = '5092-1.tif'

    # Loading data
    im = ImageIO.imReader(dataPath,fileName, 'tif',3)

    # run detection in parallel
    mc_Lists = []   
    pool = Pool(processes=1)
    params =[(i,im.data[i]) for i in range(im.size_2)]
    mc_Lists = pool.map(mc.parallelWrapper,params)
    
    global_id = mc.MC_connect_3d(mc_Lists)
    gloabal_list = mc.MCs_constuct_3d(mc_Lists,global_id)
    MC_List_3D = mc.MCs_constrain(gloabal_list)

    for item in MC_List_3D:
        print(item.center, item.intensity, item.volume)

if __name__ == '__main__':

    test_func()
