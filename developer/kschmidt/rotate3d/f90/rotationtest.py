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
for N in Ns:
   gr = gr0.copy()
   gi = gi0.copy()
   datar = makegaussians(wr,gr,sigma,N)
   data = datar+1j*makegaussians(wi,gi,sigma,N)
   
   print("Python complex128",N)
   gr = gr0.copy()
   gi = gi0.copy()
   r3df = rotate3Dpy(data)
   dmax = np.max(np.abs(data))
   for ir in Rs:
      gr = ir.apply(gr)
      gi = ir.apply(gi)
      r3df.rotation(ir)
      print("Error",np.max(np.abs(r3df.f-makegaussians(wr,gr,sigma,N)\
         -1j*makegaussians(wi,gi,sigma,N)))/dmax)
   
   print("vkfft complex128",N)
   gr = gr0.copy()
   gi = gi0.copy()
   r3df = rotate3Dv(data)
   dmax = np.max(np.abs(data))
   for ir in Rs:
      gr = ir.apply(gr)
      gi = ir.apply(gi)
      r3df.rotation(ir)
      print("Error",np.max(np.abs(r3df.f-makegaussians(wr,gr,sigma,N)\
         -1j*makegaussians(wi,gi,sigma,N)))/dmax)

   print("python legacy complex128",N)
   gr = gr0.copy()
   gi = gi0.copy()
   r3df = rotate3Dl(data)
   dmax = np.max(np.abs(data))
   for ir in Rs:
      gr = ir.apply(gr)
      gi = ir.apply(gi)
      r3df.rotation(ir)
      print("Error",np.max(np.abs(r3df.f-makegaussians(wr,gr,sigma,N)\
         -1j*makegaussians(wi,gi,sigma,N)))/dmax)
   
   print("joe complex128",N)
   gr = gr0.copy()
   gi = gi0.copy()
   dmax = np.max(np.abs(data))
   datan = data.copy()
   for ir in Rs:
      gr = ir.apply(gr)
      gi = ir.apply(gi)
      datan = rotate3Dj(datan,ir)
      print("Error",np.max(np.abs(datan-makegaussians(wr,gr,sigma,N)\
         -1j*makegaussians(wi,gi,sigma,N)))/dmax)
   
   print("Python complex64",N)
   gr = gr0.copy()
   gi = gi0.copy()
   r3df = rotate3Dpy(data.astype(np.complex64))
   dmax = np.max(np.abs(data))
   for ir in Rs:
      gr = ir.apply(gr)
      gi = ir.apply(gi)
      r3df.rotation(ir)
      print("Error",np.max(np.abs(r3df.f-makegaussians(wr,gr,sigma,N)\
         -1j*makegaussians(wi,gi,sigma,N)))/dmax)
   
   print("vkfft complex64",N)
   gr = gr0.copy()
   gi = gi0.copy()
   r3df = rotate3Dv(data.astype(np.complex64))
   dmax = np.max(np.abs(data))
   for ir in Rs:
      gr = ir.apply(gr)
      gi = ir.apply(gi)
      r3df.rotation(ir)
      print("Error",np.max(np.abs(r3df.f-makegaussians(wr,gr,sigma,N)\
         -1j*makegaussians(wi,gi,sigma,N)))/dmax)
   
   print("Python float64",N)
   gr = gr0.copy()
   gi = gi0.copy()
   r3df = rotate3Dpy(datar)
   dmax = np.max(np.abs(datar))
   for ir in Rs:
      gr = ir.apply(gr)
      r3df.rotation(ir)
      print("Error",np.max(np.abs(r3df.f-makegaussians(wr,gr,sigma,N)))/dmax)
   
   print("vkfft float64",N)
   gr = gr0.copy()
   gi = gi0.copy()
   r3df = rotate3Dv(datar)
   dmax = np.max(np.abs(datar))
   for ir in Rs:
      gr = ir.apply(gr)
      r3df.rotation(ir)
      print("Error",np.max(np.abs(r3df.f-makegaussians(wr,gr,sigma,N)))/dmax)
   
   print("Python float32",N)
   gr = gr0.copy()
   gi = gi0.copy()
   r3df = rotate3Dpy(datar.astype(np.float32))
   dmax = np.max(np.abs(datar))
   for ir in Rs:
      gr = ir.apply(gr)
      r3df.rotation(ir)
      print("Error",np.max(np.abs(r3df.f-makegaussians(wr,gr,sigma,N)))/dmax)
   
   print("vkfft float32",N)
   gr = gr0.copy()
   gi = gi0.copy()
   r3df = rotate3Dv(datar.astype(np.float32))
   dmax = np.max(np.abs(datar))
   for ir in Rs:
      gr = ir.apply(gr)
      r3df.rotation(ir)
      print("Error",np.max(np.abs(r3df.f-makegaussians(wr,gr,sigma,N)))/dmax)
