import sys
sys.path.append("..")
import numpy as np
import matplotlib.pyplot as plt

import pydiffract as dif
import pydiffract.simulate.clcore as clcore


pl = dif.detector.panelList()

pl.simpleSetup(1000,1000,100e-6,0.05,1.5e-10)

Q = pl.Q

r = np.zeros([3,3])
r[1,0] = 10e-10
r[1,1] = 1e-10

N = 5000
r = np.random.normal(0,10e-10,[N,3])

f = np.random.random([N])

A = clcore.phaseFactor(Q,r,f)

I = np.abs(A)**2
I = I.reshape((pl[0].nS, pl[0].nF))

#I[0,0:10] = I.max()*10


plt.imshow(np.log(I+1),interpolation='nearest',cmap='gray',origin='lower')
plt.show()