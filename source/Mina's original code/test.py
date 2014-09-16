from scipy import misc
import matplotlib.pyplot as plt
import mahotas
import numpy as np

import waveletcLBP as wlbp

l = misc.lena()

# testing waveletLBP
matrix = np.random.randint(100, size=(36, 98))
feats = mahotas.features.lbp(matrix,1,8,False)

feat = wlbp.concatinationhist(l)

# testing 








