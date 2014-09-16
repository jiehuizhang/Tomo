import numpy as np
from pyentropy import DiscreteSystem
import time
import MI

x = np.random.random_integers(0,19,1000)
y = x.copy()

indx = np.random.permutation(len(x))[:len(x)/2]
y[indx] = np.random.random_integers(0,9,len(x)/2)

s = DiscreteSystem(x,(1,20), y,(1,20))
s.calculate_entropies(method='plugin', calc=['HX', 'HXY'])
print s.I()


print MI.MI_init(x, y, 20,20)

