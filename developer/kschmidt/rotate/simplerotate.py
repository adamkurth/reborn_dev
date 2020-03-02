from scipy import misc
import numpy as np
import matplotlib.pyplot as plt

def shiftx(f,scale,N):
   y0 = 0.5*(N-1)
   kfac = np.zeros((N,N),dtype=np.complex128)
   xfac = np.zeros((N,N),dtype=np.complex128)
   for i in range(N):
      kfac[:,i] = np.exp(-1j*2.0*np.pi/N*np.arange(N)*scale*(i-y0))
      xfac[:,i] = np.exp(-1j*np.pi*(1-(N%2)/N)*(np.arange(N)-scale*(i-y0)))
   fk = np.fft.fft(f,axis=0)
   fk = np.fft.fftshift(fk,axes=0)
   fk *= kfac
   fs = np.fft.ifft(fk,axis=0)
   fs *= xfac
   return fs

def shifty(f,scale,N):
   x0 = 0.5*(N-1)
   kfac = np.zeros((N,N),dtype=np.complex128)
   yfac = np.zeros((N,N),dtype=np.complex128)
   for i in range(N):
      kfac[i,:] = np.exp(-1j*2.0*np.pi/N*np.arange(N)*scale*(i-x0))
      yfac[i,:] = np.exp(-1j*np.pi*(1-(N%2)/N)*(np.arange(N)-scale*(i-x0)))
   fk = np.fft.fft(f,axis=1)
   fk = np.fft.fftshift(fk,axes=1)
   fk *= kfac
   fs = np.fft.ifft(fk,axis=1)
   fs *= yfac
   return fs

def shiftrotate(f,ang,N):
   assert(np.abs(ang) < 0.26*np.pi)
   t = -np.tan(0.5*ang)
   s = np.sin(ang)
   fr = shiftx(f,t,N)
   fr = shifty(fr,s,N)
   fr = shiftx(fr,t,N)
   return fr

fin = misc.imread('bw.pgm').astype(np.float64)
assert(fin.shape[0] == fin.shape[1])
N0 = fin.shape[0]
Nadd = int(np.ceil(np.sqrt(2)*N0/2)-N0/2)
Nadd += 10
N = N0+2*Nadd
print("N",N)
fr = np.zeros((N,N),dtype=np.float64)
fr[Nadd:Nadd+N0,Nadd:Nadd+N0] = fin
f0 = fr.copy()
plt.imshow(fr,cmap=plt.cm.gray)
plt.show()
Nang = 20 
dphi = 2.0*np.pi/Nang
fr = shiftx(fr,-np.tan(0.5*dphi),N)
plt.imshow(np.real(fr),cmap=plt.cm.gray)
plt.show()
fr = shifty(fr,np.sin(dphi),N)
plt.imshow(np.real(fr),cmap=plt.cm.gray)
plt.show()
fr = shiftx(fr,-np.tan(0.5*dphi),N)
plt.imshow(np.real(fr),cmap=plt.cm.gray)
plt.show()
for i in range(Nang-1):
   fr = shiftrotate(fr,dphi,N)
   plt.imshow(np.real(fr),cmap=plt.cm.gray)
   plt.show()
