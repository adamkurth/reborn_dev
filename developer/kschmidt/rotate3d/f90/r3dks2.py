"""
Some environment variables that affect the behavior of this module:
PYOPENCL_CTX: This sets the device and platform automatically.
"""
import numpy as np
import scipy
import scipy.fftpack as fft
import pyopencl as cl
import pyopencl.array
import pyvkfft.opencl
import reborn
import reborn.simulate
import reborn.simulate.clcore


class rotate3D:

   def __init__(self,f3d):
#check f3d is 3d, cubic, double complex
      if len(f3d.shape) != 3:
         raise ValueError("rotate3D: f3d must be 3 dimensional")
      self.N = int(f3d.shape[0])
      if f3d.shape.count(self.N) != 3:
         raise ValueError("rotate3D: f3d must have all dimensions equal")
      if f3d.dtype != np.complex128:
         raise ValueError("rotate3D: f3d must be complex128")
      self.f = f3d.copy()

   def rotation(self,R):
      euler = R.as_euler('xyx')
      self.rotate3Dx(euler[0])
      self.rotate3Dy(euler[1])
      self.rotate3Dx(euler[2])

   def rotate3Dx(self,ang):
      ang=-ang
      n90 = np.rint(ang*2.0/np.pi)
      dang = ang-n90*np.pi*0.5
      n90 = int(n90 % 4)
      scale0  = -np.tan(0.5*dang)
      x0, k0, x3 = self.__getmultipliers(scale0)
      scale1 = np.sin(dang)
      x1, k1, x2 = self.__getmultipliers(scale1)
      x1 = np.transpose(x1)*x3
      k1 = np.transpose(k1).copy()
      x2 = np.transpose(x2)*x0
      for i in range(self.N):
         ftmp = self.f[i,:,:]
         if (n90 == 1):
            ftmp = np.rot90(ftmp,1,axes=(1,0))
         if (n90 == 2):
            ftmp = np.rot90(ftmp,2,axes=(1,0))
         if (n90 == 3):
            ftmp = np.rot90(ftmp,-1,axes=(1,0))
         ftmp = fft.ifft(fft.fft(x0*ftmp,axis=1)*k0,axis=1)
         ftmp = fft.ifft(fft.fft(x1*ftmp,axis=0)*k1,axis=0)
         ftmp = x3*fft.ifft(fft.fft(x2*ftmp,axis=1)*k0,axis=1)
         self.f[i,:,:] = ftmp

   def rotate3Dy(self,ang):
      n90 = np.rint(ang*2.0/np.pi)
      dang = ang-n90*np.pi*0.5
      n90 = int(n90 % 4)
      scale0  = -np.tan(0.5*dang)
      x0, k0, x3 = self.__getmultipliers(scale0)
      scale1 = np.sin(dang)
      x1, k1, x2 = self.__getmultipliers(scale1)
      x1 = np.transpose(x1)*x3
      k1 = np.transpose(k1).copy()
      x2 = np.transpose(x2)*x0
      for i in range(self.N):
         ftmp = self.f[:,i,:]
         if (n90 == 1):
            ftmp = np.rot90(ftmp,1,axes=(1,0))
         if (n90 == 2):
            ftmp = np.rot90(ftmp,2,axes=(1,0))
         if (n90 == 3):
            ftmp = np.rot90(ftmp,-1,axes=(1,0))
         ftmp = fft.ifft(fft.fft(x0*ftmp,axis=1)*k0,axis=1)
         ftmp = fft.ifft(fft.fft(x1*ftmp,axis=0)*k1,axis=0)
         ftmp = x3*fft.ifft(fft.fft(x2*ftmp,axis=1)*k0,axis=1)
         self.f[:,i,:] = ftmp

   def __getmultipliers(self,scale):
      c0 = 0.5*(self.N-1)
      nint = np.arange(self.N)
      ck = -1j*2.0*np.pi/self.N*scale
      k = np.exp(ck*(nint-c0))
      k0 = np.ones((self.N,self.N),dtype=np.complex128)
      for i in range(1,self.N):
         k0[i,:] = k0[i-1,:]*k
      k0 = np.transpose(k0).copy()
      c = -1j*np.pi*(1-(self.N%2)/self.N)
      x0 = np.tile(np.exp(c*nint),(self.N,1))
      x2 = np.tile(np.exp(-c*(nint-c0)*scale),(self.N,1))
      x1 = np.transpose(x2)*x0
      return x0,k0,x1

class rotate3Dvkfft(rotate3D):

   def __init__(self,f3d):
      super().__init__(rotate3D,self).__init__(f3d)
      vkfft_primes = (2,3,5,7,11,13)
      modprimes = self.N
      for i in range(len(vkfft_primes)):
         while modprimes % vkfft_primes[i] == 0:
            modprimes /= vkfft_primes[i]
         if modprimes == 1:
            break
      if modprimes != 1:
         raise ValueError("rotate3D: N must be a product of 2,3,5,7,11,13")
      self.ctx = reborn.simulate.clcore.create_some_gpu_context()
      self.q = cl.CommandQueue(self.ctx)
      self.mem_pool = cl.tools.MemoryPool(cl.tools.ImmediateAllocator(self.q))
#stupid complex transpose for pyvkfft on gpu -- improve me
      prg = cl.Program(ctx, """
         __kernel void transpose( __global double *at, __global double *a,
            unsigned n) {
            int ii = 2*(get_global_id(0)+get_global_id(1)*n);
            int io = 2*(get_global_id(1)+get_global_id(0)*n);
            at[io] = a[ii];
            at[io+1] = a[ii+1];
         }
         """).build ()

   def rotate3Dx(self,ang):
      ang=-ang
      n90 = np.rint(ang*2.0/np.pi)
      dang = ang-n90*np.pi*0.5
      n90 = int(n90 % 4)
      scale0  = -np.tan(0.5*dang)
      x0, k0, x3 = self.__getmultipliers(scale0)
      scale1 = np.sin(dang)
      x1, k1, x2 = self.__getmultipliers(scale1)
      x1 = np.transpose(x1)*x3
      x2 = np.transpose(x2)*x0
      x0_dev = cl.array.to_device(self.q,x0,allocator=self.mem_pool)
      for i in range(self.N):
         ftmp = self.f[i,:,:]
         if (n90 == 1):
            ftmp = np.rot90(ftmp,1,axes=(1,0))
         if (n90 == 2):
            ftmp = np.rot90(ftmp,2,axes=(1,0))
         if (n90 == 3):
            ftmp = np.rot90(ftmp,-1,axes=(1,0))

         ftmp = fft.ifft(fft.fft(x0*ftmp,axis=1)*k0,axis=1)
         ftmp = fft.ifft(fft.fft(x1*ftmp,axis=0)*k1,axis=0)
         ftmp = x3*fft.ifft(fft.fft(x2*ftmp,axis=1)*k0,axis=1)

         self.f[i,:,:] = ftmp
