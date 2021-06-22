import numpy as np
#import numpy.fft as fft
#import pyfftw.interfaces.numpy_fft as fft #slowest
import scipy.fftpack as fft #fastest

def rotate3Dy(f,dang):
   N = np.max(f.shape)
   n90 = int(np.rint(dang*2.0/np.pi))
   dang = dang-n90*np.pi*0.5
   n90 = int(n90 % 4)
   if (n90 == 1):
      f = np.rot90(f,1,axes=(0,2))
   if (n90 == 2):
      f = np.rot90(f,2,axes=(0,2))
   if (n90 == 3):
      f = np.rot90(f,-1,axes=(0,2))
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
   x1t = np.transpose(x1)

   for i in range(N):
      ftmp = f[:,i,:]
      ftmp = zfac*fft.ifft(fft.fft(x1t*ftmp,axis=0)*kfacz,axis=0)
      ftmp = xfac*fft.ifft(fft.fft(ftmp*z1,axis=1)*kfacx,axis=1)
      ftmp = zfac*fft.ifft(fft.fft(x1t*ftmp,axis=0)*kfacz,axis=0)
      f[:,i,:] = ftmp

   return f

def rotate3Dz(f,dang):
   N = np.max(f.shape)
   n90 = np.rint(dang*2.0/np.pi)
   dang = dang-n90*np.pi*0.5
   n90 = int(n90 % 4)
   if (n90 == 1):
      f = np.rot90(f,1,axes=(2,1))
   if (n90 == 2):
      f = np.rot90(f,2,axes=(2,1))
   if (n90 == 3):
      f = np.rot90(f,-1,axes=(2,1))
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
   y1t = np.transpose(y1)
   yfac = y2*y1t

   for i in range(N):
      ftmp = f[i,:,:]
      ftmp = xfac*fft.ifft(fft.fft(x1*ftmp,axis=1)*kfacx,axis=1)
      ftmp = yfac*fft.ifft(fft.fft(y1t*ftmp,axis=0)*kfacy,axis=0)
      ftmp = xfac*fft.ifft(fft.fft(x1*ftmp,axis=1)*kfacx,axis=1)
      f[i,:,:] = ftmp

   return f

def rotate3Dks(f, euler):
# f is assumed f(z,y,x)
# Euler angles are around lab z, lab y, lab z
   f = rotate3Dz(f,euler[0])
   f = rotate3Dy(f,euler[1])
   f = rotate3Dz(f,euler[2])
   return f
