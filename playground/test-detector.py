import sys
sys.path.append("..")
import time
import numpy as np
import matplotlib.pyplot as plt

import bornagain as dif
import bornagain.simulate.clcore as clcore


pl = dif.detector.panelList()

pl.simpleSetup(1000,1000,100e-6,0.05,1.5e-10)

Q = pl.Q

# r = np.zeros([3,3])
# 
# r[1,0] = 0e-10
# r[1,1] = -3e-10
# r[1,2] = 0e-10
# 
# r[2,0] = 6e-10
# r[2,1] = 3e-10

N = 10000
r = np.random.normal(0,10e-10,[N,3])
#r[:,2] = 0

f = np.random.random([N]) + 1j*np.random.random([N])
#f = None

t = time.time()
A = clcore.phaseFactor(Q,r,f)
print(time.time() - t)

I = np.abs(A)**2
I = I.reshape((pl[0].nS, pl[0].nF))

#I[0,0:10] = I.max()*10

plt.imshow(np.log(I+1),interpolation='nearest',cmap='gray',origin='lower')
plt.title('y: up, x: right, z: beam (towards you)')
plt.show()