import numpy as np
#import numpy.fft as fft
#import pyfftw.interfaces.numpy_fft as fft #slowest
import scipy.fftpack as fft #fastest

def rotate3Dy(f,dang):
   N = np.max(f.shape)
   n90 = np.rint(dang*2.0/np.pi)
   dang = dang-n90*np.pi*0.5
   if (n90 % 4 == 1):
      f = np.transpose(np.flip(f,axis=0),axes=(2,1,0))
   if (n90 % 4 == 2):
      f = np.flip(f,axis=(0,2))
   if (n90 % 4 == 3):
      f = np.transpose(np.flip(f,axis=2),axes=(2,1,0))
   scalez = -np.tan(0.5*dang)
   scalex = np.sin(dang)
   c0 = 0.5*(N-1)
   nint = np.arange(N)

   ck = -1j*2.0*np.pi/N*scalez
   k1 = np.exp(ck*(nint-c0))
   kfacz = np.ones((N,N),dtype=np.complex128)
   for i in range(1,N):
      kfacz[i,:] = kfacz[i-1,:]*k1
   c = -1j*np.pi*(1-(N%2)/N)
   z1 = np.tile(np.exp(c*nint),(N,1))
   z2 = np.tile(np.exp(-c*(nint-c0)*scalez),(N,1))
   zfac = np.transpose(z1)*z2

   ck = -1j*2.0*np.pi/N*scalex
   kfacx = np.zeros((N,N),dtype=np.complex128)
   k1 = np.exp(ck*nint)
   kfacx[0,:] = np.exp(-ck*nint*c0)
   for i in range(1,N):
      kfacx[i,:] = kfacx[i-1,:]*k1
   cx = -1j*np.pi*(1-(N%2)/N)
   x1 = np.tile(np.exp(cx*nint),(N,1))
   x2 = np.tile(np.exp(-cx*(nint-c0)*scalex),(N,1))
   xfac = x1*np.transpose(x2)

   for i in range(N):
      ftmp = f[:,i,:]
      ftmp = fft.fftshift(fft.fft(ftmp,axis=0),axes=0)
      ftmp *= kfacz
      ftmp = fft.ifft(ftmp,axis=0)
      ftmp *= zfac

      ftmp = fft.fftshift(fft.fft(ftmp,axis=1),axes=1)
      ftmp *= kfacx
      ftmp = fft.ifft(ftmp,axis=1)
      ftmp *= xfac

      ftmp = fft.fftshift(fft.fft(ftmp,axis=0),axes=0)
      ftmp *= kfacz
      ftmp = fft.ifft(ftmp,axis=0)
      ftmp *= zfac
      f[:,i,:] = ftmp

   return f

def rotate3Dz(f,dang):
   N = np.max(f.shape)
   n90 = np.rint(dang*2.0/np.pi)
   dang = dang-n90*np.pi*0.5
   if (n90 % 4 == 1):
      f = np.transpose(np.flip(f,axis=1),axes=(1,0,2))
   if (n90 % 4 == 2):
      f = np.flip(f,axis=(0,1))
   if (n90 % 4 == 3):
      f = np.transpose(np.flip(f,axis=0),axes=(1,0,2))
   scalex  = -np.tan(0.5*dang)
   scaley = np.sin(dang)
   c0 = 0.5*(N-1)
   nint = np.arange(N)
   ck = -1j*2.0*np.pi/N*scalex
   k1 = np.exp(ck*(nint-c0))
   kfacx = np.ones((N,N),dtype=np.complex128)
   for i in range(1,N):
      kfacx[i,:] = kfacx[i-1,:]*k1
   kfacx = np.transpose(kfacx)
   c = -1j*np.pi*(1-(N%2)/N)
   x1 = np.tile(np.exp(c*nint),(N,1))
   x2 = np.tile(np.exp(-c*(nint-c0)*scalex),(N,1))
   xfac = np.transpose(x2)*x1

   ck = -1j*2.0*np.pi/N*scaley
   kfacy = np.zeros((N,N),dtype=np.complex128)
   k1 = np.exp(ck*nint)
   kfacy[0,:] = np.exp(-ck*nint*c0)
   for i in range(1,N):
      kfacy[i,:] = kfacy[i-1,:]*k1
   kfacy = np.transpose(kfacy)
   cy = -1j*np.pi*(1-(N%2)/N)
   y1 = np.tile(np.exp(cy*nint),(N,1))
   y2 = np.tile(np.exp(-cy*(nint-c0)*scaley),(N,1))
   yfac = y2*np.transpose(y1)

   for i in range(N):
      ftmp = f[i,:,:]
      ftmp = fft.fftshift(fft.fft(ftmp,axis=1),axes=1)
      ftmp *= kfacx
      ftmp = fft.ifft(ftmp,axis=1)
      ftmp *= xfac

      ftmp = fft.fftshift(fft.fft(ftmp,axis=0),axes=0)
      ftmp *= kfacy
      ftmp = fft.ifft(ftmp,axis=0)
      ftmp *= yfac

      ftmp = fft.fftshift(fft.fft(ftmp,axis=1),axes=1)
      ftmp *= kfacx
      ftmp = fft.ifft(ftmp,axis=1)
      ftmp *= xfac
      f[i,:,:] = ftmp

   return f

def rotate3Dks(f, euler):
# f is assumed f(z,y,x)
# Euler angles are around lab z, lab y, lab z
   f = rotate3Dz(f,euler[0])
   f = rotate3Dy(f,euler[1])
   f = rotate3Dz(f,euler[2])
   return f

if __name__ == "__main__":
   import time
   import matplotlib.pyplot as plt
   from reborn.utils import rotate3D
   
   fin = plt.imread('bw4.png').astype(np.float64)
   assert(fin.shape[0] == fin.shape[1])
   N0 = fin.shape[0]
   Nadd = int(np.ceil(np.sqrt(2)*N0/2)-N0/2)
   Nadd += 10
   N = N0+2*Nadd
   print("N",N)
   fr = np.zeros((N,N),dtype=np.float64)
   fr[Nadd:Nadd+N0,Nadd:Nadd+N0] = fin
   f3d = np.zeros((N,N,N),dtype=np.complex128)
   for i in range(N):
      f3d[i,:,:] = np.transpose(fr)
   
   euler = (0.1, 0.17, 0.35)
   eulerj = (-0.1, 0.17, -0.35)
# For these to be identical, with no negative signs, change joe's code to
# f_rot[ii, :, :] = np.transpose(rotate2D(np.transpose(f[ii, :, :]), kxfac, xfac, kyfac, yfac, n90_mod_Four))
   
   t1 = time.time()
   f1 = rotate3D(f3d,eulerj)
   t2 = time.time()
   f2 = rotate3Dks(f3d,euler)
   t3 = time.time()
   print("original",f3d[50,50,50])
   print(f1[50,50,50],f2[50,50,50])
   print("joe ",t2-t1,"seconds")
   print("kevin ",t3-t2,"seconds")
   
   print("joe rotation")
   plt.imshow(np.real(f1[0,:,:]),cmap=plt.cm.gray)
   plt.show()
   print("kevin rotation")
   plt.imshow(np.real(f2[0,:,:]),cmap=plt.cm.gray)
   plt.show()
