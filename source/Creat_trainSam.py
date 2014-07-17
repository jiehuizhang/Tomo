""" Creat Training Samples from image crop"""

import sys, os, re, operator, pdb, subprocess, time
import numpy as np
import ImageIO
import TPatch

def creatTrainigSam(dataPath,numrings = 10):

    patches_feats = np.zeros((1,2*numrings), dtype=np.double)
    file_list = os.listdir(dataPath)
    for fil in file_list:
        im = ImageIO.imReader(dataPath, fil,'tif',2)

        patch = TPatch.TPatch()
        patch.initialize(im.data[0])
        patch.getRings(numrings)
        patch.getMeanFeats()
        patch.getVarFeats()
        patches_feats = np.vstack((patches_feats,patch.dumpFeats()))

    return patches_feats
