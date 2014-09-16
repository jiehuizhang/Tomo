
from __future__ import division
import numpy as np
from MI_utils import (prob, _probcount, decimalise, pt_bayescount, 
                   nsb_entropy, dec2base, ent, malog2)

def MI(PX, PY):

    mask1 = PX > 0
    mask2 = PY > 0
    mask = mask1 * mask2
    PX = PX[mask]
    PY = PY[mask]
    kbp = np.sum(PX * np.log(PX / PY), 0)
    kbq = np.sum(PY * np.log(PY / PX), 0)
    
    return np.double(kbp + kbq)/2






