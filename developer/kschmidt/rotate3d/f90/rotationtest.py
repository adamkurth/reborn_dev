import time
import numpy as np
import scipy
import scipy.spatial
from r3dks4 import rotate3D as rotate3Dpy
from r3dks4 import rotate3Dvkfft as rotate3Dv
from r3dks4 import rotate3Dlegacy as rotate3Dl
from reborn.utils import rotate3D as rotate3Dj
import matplotlib.pyplot as plt


def makegaussians(w,g,s,N):
   x = np.linspace(-0.5*(1.0-1.0/N),0.5*(1.0-1.0/N),num=N)
   d = np.zeros((N,N,N),dtype=np.float64)
   for ig in range(len(w)):
      xgauss = np.exp(-0.5*((x-g[ig,0])/s)**2)
      ygauss = np.exp(-0.5*((x-g[ig,1])/s)**2)
      zgauss = np.exp(-0.5*((x-g[ig,2])/s)**2)
      d += w[ig]*np.tile(np.reshape(xgauss,(N,1,1)),(1,N,N))*\
         np.tile(np.reshape(ygauss,(1,N,1)),(N,1,N))*\
         np.tile(np.reshape(zgauss,(1,1,N)),(N,N,1))
   return d

rng = np.random.default_rng(1717171717)
Nr = 10
Rs = scipy.spatial.transform.Rotation.random(Nr,random_state=rng)
Ngr = 8 
Ngi = 8
sigma = 0.05
gr0 = (rng.random((Ngr,3))-0.5)*sigma
wr = rng.random(Ngr)-0.5
gi0 = (rng.random((Ngi,3))-0.5)*sigma
wi = rng.random(Ngi)-0.5
Ns = [7, 16, 27, 32, 48, 64, 75]
methods = [rotate3Dpy, rotate3Dv, rotate3Dl, rotate3Dj]
types = [np.complex128, np.complex64, np.float64, np.float32]
for N in Ns:
   gr = gr0.copy()
   gi = gi0.copy()
   datar = makegaussians(wr,gr,sigma,N)
   data = datar+1j*makegaussians(wi,gi,sigma,N)
   
   for t in types:
      for m in methods:
         dmax = np.max(np.abs(data))
         if m == rotate3Dj:
            if t == np.complex128:
               print("Joe " + t.__name__,N)
               gr = gr0.copy()
               gi = gi0.copy()
               datan = data.copy()
               for ir in Rs:
                  gr = ir.apply(gr)
                  gi = ir.apply(gi)
                  datan = rotate3Dj(datan,ir)
                  print("Error",np.max(np.abs(datan\
                     -makegaussians(wr,gr,sigma,N)\
                     -1j*makegaussians(wi,gi,sigma,N)))/dmax)
               continue
            continue
         if m == rotate3Dl and t != np.complex128:
            continue
         print(m.__name__ + " " + t.__name__,N)
         gr = gr0.copy()
         gi = gi0.copy()
         if t == np.complex128 or t == np.complex64:
            r3df = m(data.astype(t))
            dmax = np.max(np.abs(data))
         else:
            r3df = m(datar.astype(t))
            dmax = np.max(np.abs(datar))
         for ir in Rs:
            gr = ir.apply(gr)
            gi = ir.apply(gi)
            r3df.rotation(ir)
            if t == np.complex128 or t == np.complex64:
               print("Error",np.max(np.abs(r3df.f-makegaussians(wr,gr,sigma,N)\
                  -1j*makegaussians(wi,gi,sigma,N)))/dmax)
            else:
               print("Error",np.max(np.abs(r3df.f-makegaussians(wr,gr,sigma,N)\
                  ))/dmax)
