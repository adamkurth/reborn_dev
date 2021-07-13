"""
Some environment variables that affect the behavior of this module:
PYOPENCL_CTX: This sets the device and platform automatically.
"""
import numpy as np
import scipy
import scipy.fftpack as fft



class rotate3D:
   r"""
    Base class to rotate a 3D array of double precision complex numbers in
    3-dimensions.  The function works by rotating each 2D sections of
    the 3D array via three shears, as described by Unser et al. (1995)
    "Convolution-based interpolation for fast, high-quality rotation of images."
    IEEE Transactions on Image Processing, 4:1371.

    Note 1: The input array must be 3d, double complex, and have all three
            dimension sizes equal. Otherwise it raises a ValueError exception.

    Note 2: If you don't want wrap arounds, make sure the input array, f,
            is zero-padded to at least sqrt(2) times the largest dimension
            of the desired object.

    Arguments:
        f (*3D |ndarray|*) : The 3D input array. f is the corresponding
        class member for output.

    Methods:
      rotation(R): R is a rotation specified as a
                   scipy.spatial.transform.Rotation 
   """

   def __init__(self,f3d):
      self.N = 0
      self.f = f3d

   @property
   def f(self):
      return self._f

   @f.setter
   def f(self,f):
      self._checkf(f)
      self._f = f.copy()

   def _checkf(self,f3d):
#check f3d is 3d, cubic, double complex
      if len(f3d.shape) != 3:
         raise ValueError("rotate3D: f3d must be 3 dimensional")
      if self.N == 0:
         self.N = int(f3d.shape[0])
      if f3d.shape.count(self.N) != 3:
         if self.N == 0:
            raise ValueError("rotate3D: f3d must have all dimensions equal")
         else:
            raise ValueError(
               "rotate3D: f3d must have all dimensions equal to N")
      if f3d.dtype != np.complex128:
         raise ValueError("rotate3D: f3d must be complex128")

   def rotation(self,R):
      euler = R.as_euler('xyx')
      self._rotate3Dx(euler[0])
      self._rotate3Dy(euler[1])
      self._rotate3Dx(euler[2])

   def _rotate3Dx(self,angin):
# angle negative since done in order z then y
      ang = -angin
      n90 = np.rint(ang*2.0/np.pi)
      dang = ang-n90*np.pi*0.5
      n90 = int(n90 % 4)
      scale0  = -np.tan(0.5*dang)
      x0, k0, x3 = self._getmultipliers(scale0)
      scale1 = np.sin(dang)
      x1, k1, x2 = self._getmultipliers(scale1)
      x1 = np.transpose(x1)*x3
      k1 = np.transpose(k1).copy()
      x2 = np.transpose(x2)*x0
      for i in range(self.N):
         ftmp = self._f[i,:,:]
         if (n90 == 1):
            ftmp = np.rot90(ftmp,1,axes=(1,0))
         if (n90 == 2):
            ftmp = np.rot90(ftmp,2,axes=(1,0))
         if (n90 == 3):
            ftmp = np.rot90(ftmp,-1,axes=(1,0))
         ftmp = fft.ifft(fft.fft(x0*ftmp,axis=1)*k0,axis=1)
         ftmp = fft.ifft(fft.fft(x1*ftmp,axis=0)*k1,axis=0)
         ftmp = x3*fft.ifft(fft.fft(x2*ftmp,axis=1)*k0,axis=1)
         self._f[i,:,:] = ftmp

   def _rotate3Dy(self,ang):
# identical to x rotation except angle positive since
# this is done in the order z then x, and the array slices
# in for loop are x-z.
      
      n90 = np.rint(ang*2.0/np.pi)
      dang = ang-n90*np.pi*0.5
      n90 = int(n90 % 4)
      scale0  = -np.tan(0.5*dang)
      x0, k0, x3 = self._getmultipliers(scale0)
      scale1 = np.sin(dang)
      x1, k1, x2 = self._getmultipliers(scale1)
      x1 = np.transpose(x1)*x3
      k1 = np.transpose(k1).copy()
      x2 = np.transpose(x2)*x0
      for i in range(self.N):
         ftmp = self._f[:,i,:]
         if (n90 == 1):
            ftmp = np.rot90(ftmp,1,axes=(1,0))
         if (n90 == 2):
            ftmp = np.rot90(ftmp,2,axes=(1,0))
         if (n90 == 3):
            ftmp = np.rot90(ftmp,-1,axes=(1,0))
         ftmp = fft.ifft(fft.fft(x0*ftmp,axis=1)*k0,axis=1)
         ftmp = fft.ifft(fft.fft(x1*ftmp,axis=0)*k1,axis=0)
         ftmp = x3*fft.ifft(fft.fft(x2*ftmp,axis=1)*k0,axis=1)
         self._f[:,i,:] = ftmp

   def _getmultipliers(self,scale):
      c0 = 0.5*(self.N-1)
      nint = np.arange(self.N)
      ck = -1j*2.0*np.pi/self.N*scale
      k = np.exp(ck*(nint-c0))
      k0 = np.ones((self.N,self.N),dtype=np.complex128)
      for i in range(1,self.N):
         k0[i,:] = k0[i-1,:]*k
      k0 = np.transpose(k0).copy()
      c = 1j*np.pi*(1-(self.N%2)/self.N)
      x0 = np.tile(np.exp(c*nint),(self.N,1))
      x2 = np.tile(np.exp(c*(nint-c0)*scale),(self.N,1))
      x1 = np.transpose(x2)*np.conjugate(x0)
      return x0,k0,x1

class rotate3Djoeorder(rotate3D):
   r"""
    This is identical to rotate3D except that the shear orders are
    the same as in Joe's original code. It is somewhat less efficient
    since the arrays are transposed and then transposed back.
  """

   def __init__(self,f3d):
      super().__init__(f3d)

   def rotation(self,R):
      euler = R.as_euler('xyx')
      self._f = np.transpose(self._f,axes=(0,2,1))
      self._rotate3Dx(-euler[0])
      self._f = np.transpose(self._f,axes=(1,2,0))
      self._rotate3Dy(-euler[1])
      self._f = np.transpose(self._f,axes=(2,0,1))
      self._rotate3Dx(-euler[2])
      self._f = np.transpose(self._f,axes=(0,2,1))


import pyopencl as cl
import pyopencl.array
import pyvkfft.opencl
import reborn
import reborn.simulate
import reborn.simulate.clcore

class rotate3Dvkfft(rotate3D):
   r"""
    This should give results identical to rotate3D for sizes
    that are products of powers of 2,3,5,7,11,13.  It uses the pyvkfft
    wrapper to VkFFT to perform Fourier transforms on a gpu. Since this
    is my first opencl code, it is no doubt written inefficiently.
   """

   def __init__(self,f3d):
      super().__init__(f3d)
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
#      self.mem_pool = cl.tools.MemoryPool(cl.tools.ImmediateAllocator(self.q))
      self.mem_pool = None
      self.app = pyvkfft.opencl.VkFFTApp((self.N,self.N),
         dtype=np.complex128,queue=self.q,ndim=1)
#stupid complex transpose for pyvkfft on gpu -- improve me
      self.prg = cl.Program(self.ctx, """
         __kernel void transpose( __global double2 *at, __global double2 *a,
            unsigned n) {
            int ii = get_global_id(0)+get_global_id(1)*n;
            int io = get_global_id(1)+get_global_id(0)*n;
            at[io] = a[ii];
         }
         """).build ()
      dummy  = np.ndarray((self.N,self.N),np.complex128)
      self.x0_dev = cl.array.to_device(self.q,dummy,allocator=self.mem_pool)
      self.x1_dev = cl.array.to_device(self.q,dummy,allocator=self.mem_pool)
      self.x2_dev = cl.array.to_device(self.q,dummy,allocator=self.mem_pool)
      self.x3_dev = cl.array.to_device(self.q,dummy,allocator=self.mem_pool)
      self.k0_dev = cl.array.to_device(self.q,dummy,allocator=self.mem_pool)
      self.k1_dev = cl.array.to_device(self.q,dummy,allocator=self.mem_pool)
      self.f_dev = cl.array.to_device(self.q,dummy,allocator=self.mem_pool)
      self.ft_dev = cl.array.to_device(self.q,dummy,allocator=self.mem_pool)

   def _rotate3Dx(self,angin):
      ang=-angin
      n90 = np.rint(ang*2.0/np.pi)
      dang = ang-n90*np.pi*0.5
      n90 = int(n90 % 4)
      scale0  = -np.tan(0.5*dang)
      x0, k0, x3 = self._getmultipliers(scale0)
      scale1 = np.sin(dang)
      x1, k1, x2 = self._getmultipliers(scale1)
      x1 = np.transpose(x1)*x3
      x2 = np.transpose(x2)*x0
      cl.enqueue_copy(self.q,self.x0_dev.data,x0)
      cl.enqueue_copy(self.q,self.x1_dev.data,x1)
      cl.enqueue_copy(self.q,self.x2_dev.data,x2)
      cl.enqueue_copy(self.q,self.x3_dev.data,x3)
      cl.enqueue_copy(self.q,self.k0_dev.data,k0)
      cl.enqueue_copy(self.q,self.k1_dev.data,k1)
      for i in range(self.N):
         ftmp = self._f[i,:,:]
         if (n90 == 1):
            ftmp = np.rot90(ftmp,1,axes=(1,0))
         if (n90 == 2):
            ftmp = np.rot90(ftmp,2,axes=(1,0))
         if (n90 == 3):
            ftmp = np.rot90(ftmp,-1,axes=(1,0))
         ftmp = ftmp.copy()
         cl.enqueue_copy(self.q,self.f_dev.data,ftmp)
         self.f_dev = self.x0_dev*self.f_dev
         self.app.fft(self.f_dev)
         self.f_dev = self.k0_dev*self.f_dev
         self.app.ifft(self.f_dev)
         self.f_dev = self.x1_dev*self.f_dev
         self.prg.transpose(self.q,(self.N,self.N),None,self.ft_dev.data,
            self.f_dev.data,np.uint32(self.N))
         self.app.fft(self.ft_dev)
         self.ft_dev = self.k1_dev*self.ft_dev
         self.app.ifft(self.ft_dev)
         self.prg.transpose(self.q,(self.N,self.N),None,self.f_dev.data,
            self.ft_dev.data,np.uint32(self.N))
         self.f_dev = self.x2_dev*self.f_dev
         self.app.fft(self.f_dev)
         self.f_dev = self.k0_dev*self.f_dev
         self.app.ifft(self.f_dev)
         self.f_dev = self.x3_dev*self.f_dev
         cl.enqueue_copy(self.q,ftmp,self.f_dev.data)
         self._f[i,:,:] = ftmp

   def _rotate3Dy(self,ang):
# identical to x rotation except angle positive since
# this is done in the order z then x, and the array slices
# in for loop are x-z.
      n90 = np.rint(ang*2.0/np.pi)
      dang = ang-n90*np.pi*0.5
      n90 = int(n90 % 4)
      scale0  = -np.tan(0.5*dang)
      x0, k0, x3 = self._getmultipliers(scale0)
      scale1 = np.sin(dang)
      x1, k1, x2 = self._getmultipliers(scale1)
      x1 = np.transpose(x1)*x3
      x2 = np.transpose(x2)*x0
      cl.enqueue_copy(self.q,self.x0_dev.data,x0)
      cl.enqueue_copy(self.q,self.x1_dev.data,x1)
      cl.enqueue_copy(self.q,self.x2_dev.data,x2)
      cl.enqueue_copy(self.q,self.x3_dev.data,x3)
      cl.enqueue_copy(self.q,self.k0_dev.data,k0)
      cl.enqueue_copy(self.q,self.k1_dev.data,k1)
      for i in range(self.N):
         ftmp = self._f[:,i,:]
         if (n90 == 1):
            ftmp = np.rot90(ftmp,1,axes=(1,0))
         if (n90 == 2):
            ftmp = np.rot90(ftmp,2,axes=(1,0))
         if (n90 == 3):
            ftmp = np.rot90(ftmp,-1,axes=(1,0))
         ftmp = ftmp.copy()
         cl.enqueue_copy(self.q,self.f_dev.data,ftmp)
         self.f_dev = self.x0_dev*self.f_dev
         self.app.fft(self.f_dev)
         self.f_dev = self.k0_dev*self.f_dev
         self.app.ifft(self.f_dev)
         self.f_dev = self.x1_dev*self.f_dev
         self.prg.transpose(self.q,(self.N,self.N),None,self.ft_dev.data,
            self.f_dev.data,np.uint32(self.N))
         self.app.fft(self.ft_dev)
         self.ft_dev = self.k1_dev*self.ft_dev
         self.app.ifft(self.ft_dev)
         self.prg.transpose(self.q,(self.N,self.N),None,self.f_dev.data,
            self.ft_dev.data,np.uint32(self.N))
         self.f_dev = self.x2_dev*self.f_dev
         self.app.fft(self.f_dev)
         self.f_dev = self.k0_dev*self.f_dev
         self.app.ifft(self.f_dev)
         self.f_dev = self.x3_dev*self.f_dev
         cl.enqueue_copy(self.q,ftmp,self.f_dev.data)
         self._f[:,i,:] = ftmp

class rotate3Dvkfft_stored_on_device(rotate3D):
   r"""
    This should give results identical to rotate3D for sizes
    that are products of powers of 2,3,5,7,11,13.  It uses the pyvkfft
    wrapper to VkFFT to perform Fourier transforms on a gpu. Since this
    is my first opencl code, it is no doubt written inefficiently.

    Additional Argument:
       dev_dtype: defaults to np.complex128. Setting it to np.complex64
       makes the device calculations single precision.
   """

   def __init__(self,f3d,dev_dtype=np.complex128):
      self.N = 0
      self.f_dev = None
      self.dev_dtype = dev_dtype
      self._checkf(f3d)
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
      self.app = pyvkfft.opencl.VkFFTApp((self.N,self.N,self.N),
         dtype=self.dev_dtype,queue=self.q,ndim=1)
      self.factors = np.ndarray((18,self.N,self.N),self.dev_dtype)
      self.factors_dev = cl.array.to_device(self.q,self.factors)
      self.f_dev = cl.array.to_device(self.q,
         f3d.astype(self.dev_dtype,order='C'))
      #stupid routines on gpu -- improve me
      prg_double = cl.Program(self.ctx, """
         __kernel void transposeyz(  __global double2 *a, unsigned n) {
            double2 temp;
            int ii = get_global_id(0)+(get_global_id(1)+get_global_id(2)*n)*n;
            int io = get_global_id(1)+(get_global_id(0)+get_global_id(2)*n)*n;
            if (ii <= io) {
               temp = a[io];
               a[io] = a[ii];
               a[ii] = temp;
            }
         }
   
         __kernel void transposexy( __global double2 *a, unsigned n) {
            double2 temp;
            int ii = get_global_id(0)+(get_global_id(1)+get_global_id(2)*n)*n;
            int io = get_global_id(0)+(get_global_id(2)+get_global_id(1)*n)*n;
            if (ii <= io) {
               temp = a[io];
               a[io] = a[ii];
               a[ii] = temp;
            }
         }
   
         __kernel void multiply_ith( __global double2 *factor,\
             __global double2 *a, unsigned n, unsigned ith) {
            int ifac = get_global_id(0)+n*(get_global_id(1)+n*ith);
            int i = get_global_id(0)+(get_global_id(1)+get_global_id(2)*n)*n;
            double tempr = a[i].x;
            a[i].x = factor[ifac].x*a[i].x-factor[ifac].y*a[i].y;
            a[i].y = factor[ifac].x*a[i].y+factor[ifac].y*tempr;
         }

         // Call with (n/2,n/2,n)
         __kernel void rot90ev( __global double2 *a, unsigned n) {
            int n2=n/2;
            int i0 =\
               n2+get_global_id(0)+n*(n2+get_global_id(1)+n*get_global_id(2));
            int i1 =\
               n2-get_global_id(1)-1+n*(n2+get_global_id(0)+n*get_global_id(2));
            int i2 =\
               n2-get_global_id(0)-1\
               +n*(n2-get_global_id(1)-1+n*get_global_id(2));
            int i3 =\
               n2+get_global_id(1)+n*(n2-get_global_id(0)-1+n*get_global_id(2));
            double2 temp = a[i0];
            a[i0] = a[i3];
            a[i3] = a[i2];
            a[i2] = a[i1];
            a[i1] = temp;
         }

         // Call with (n/2,n/2,n)
         __kernel void rot270ev( __global double2 *a, unsigned n) {
            int n2=n/2;
            int i0 =\
               n2+get_global_id(0)+n*(n2+get_global_id(1)+n*get_global_id(2));
            int i1 =\
               n2-get_global_id(1)-1+n*(n2+get_global_id(0)+n*get_global_id(2));
            int i2 =\
               n2-get_global_id(0)-1\
               +n*(n2-get_global_id(1)-1+n*get_global_id(2));
            int i3 =\
               n2+get_global_id(1)\
               +n*(n2-get_global_id(0)-1+n*get_global_id(2));
            double2 temp = a[i0];
            a[i0] = a[i1];
            a[i1] = a[i2];
            a[i2] = a[i3];
            a[i3] = temp;
         }

         // Call with (n/2,n/2,n)
         __kernel void rot180ev( __global double2 *a, unsigned n) {
            //This could be made more efficient with (n,n/2,n) and do just
            //one interchange per item.
            int n2=n/2;
            int i0 =\
               n2+get_global_id(0)+n*(n2+get_global_id(1)+n*get_global_id(2));
            int i1 =\
               n2-get_global_id(1)-1+n*(n2+get_global_id(0)+n*get_global_id(2));
            int i2 =\
               n2-get_global_id(0)-1\
               +n*(n2-get_global_id(1)-1+n*get_global_id(2));
            int i3 =\
               n2+get_global_id(1)+n*(n2-get_global_id(0)-1+n*get_global_id(2));
            double2 temp = a[i0];
            a[i0] = a[i2];
            a[i2] = temp;
            temp = a[i1];
            a[i1] = a[i3];
            a[i3] = temp;
         }

         // Call with ((n+1)/2,(n-1)/2,n)
         __kernel void rot90odd( __global double2 *a, unsigned n) {
            int n2=(n+1)/2;
            int i0 =\
               n2-1+get_global_id(0)\
               +n*(n2+get_global_id(1)+n*get_global_id(2));
            int i1 =\
               n2-2-get_global_id(1)\
               +n*(n2-1+get_global_id(0)+n*get_global_id(2));
            int i2 =\
               n2-1-get_global_id(0)\
               +n*(n2-2-get_global_id(1)+n*get_global_id(2));
            int i3 =\
               n2+get_global_id(1)+n*(n2-1-get_global_id(0)+n*get_global_id(2));
            double2 temp = a[i0];
            a[i0] = a[i3];
            a[i3] = a[i2];
            a[i2] = a[i1];
            a[i1] = temp;
         }

         // Call with ((n+1)/2,(n-1)/2,n)
         __kernel void rot270odd( __global double2 *a, unsigned n) {
            int n2=(n+1)/2;
            int i0 =\
               n2-1+get_global_id(0)\
               +n*(n2+get_global_id(1)+n*get_global_id(2));
            int i1 =\
               n2-2-get_global_id(1)\
               +n*(n2-1+get_global_id(0)+n*get_global_id(2));
            int i2 =\
               n2-1-get_global_id(0)\
               +n*(n2-2-get_global_id(1)+n*get_global_id(2));
            int i3 =\
               n2+get_global_id(1)\
               +n*(n2-1-get_global_id(0)+n*get_global_id(2));
            double2 temp = a[i0];
            a[i0] = a[i1];
            a[i1] = a[i2];
            a[i2] = a[i3];
            a[i3] = temp;
         }

         // Call with ((n+1)/2,(n-1)/2,n)
         __kernel void rot180odd( __global double2 *a, unsigned n) {
            int n2=(n+1)/2;
            int i0 =\
               n2-1+get_global_id(0)\
               +n*(n2+get_global_id(1)+n*get_global_id(2));
            int i1 =\
               n2-2-get_global_id(1)\
               +n*(n2-1+get_global_id(0)+n*get_global_id(2));
            int i2 =\
               n2-1-get_global_id(0)
               +n*(n2-2-get_global_id(1)+n*get_global_id(2));
            int i3 =\
               n2+get_global_id(1)\
               +n*(n2-1-get_global_id(0)+n*get_global_id(2));
            double2 temp = a[i0];
            a[i0] = a[i2];
            a[i2] = temp;
            temp = a[i1];
            a[i1] = a[i3];
            a[i3] = temp;
         }

         """)

      prg_single = cl.Program(self.ctx, """
         __kernel void transposeyz(  __global float2 *a, unsigned n) {
            float2 temp;
            int ii = get_global_id(0)+(get_global_id(1)+get_global_id(2)*n)*n;
            int io = get_global_id(1)+(get_global_id(0)+get_global_id(2)*n)*n;
            if (ii <= io) {
               temp = a[io];
               a[io] = a[ii];
               a[ii] = temp;
            }
         }
   
         __kernel void transposexy( __global float2 *a, unsigned n) {
            float2 temp;
            int ii = get_global_id(0)+(get_global_id(1)+get_global_id(2)*n)*n;
            int io = get_global_id(0)+(get_global_id(2)+get_global_id(1)*n)*n;
            if (ii <= io) {
               temp = a[io];
               a[io] = a[ii];
               a[ii] = temp;
            }
         }
   
         __kernel void multiply_ith( __global float2 *factor,\
             __global float2 *a, unsigned n, unsigned ith) {
            int ifac = get_global_id(0)+n*(get_global_id(1)+n*ith);
            int i = get_global_id(0)+(get_global_id(1)+get_global_id(2)*n)*n;
            float tempr = a[i].x;
            a[i].x = factor[ifac].x*a[i].x-factor[ifac].y*a[i].y;
            a[i].y = factor[ifac].x*a[i].y+factor[ifac].y*tempr;
         }

         // Call with (n/2,n/2,n)
         __kernel void rot90ev( __global float2 *a, unsigned n) {
            int n2=n/2;
            int i0 =\
               n2+get_global_id(0)+n*(n2+get_global_id(1)+n*get_global_id(2));
            int i1 =\
               n2-get_global_id(1)-1+n*(n2+get_global_id(0)+n*get_global_id(2));
            int i2 =\
               n2-get_global_id(0)-1\
               +n*(n2-get_global_id(1)-1+n*get_global_id(2));
            int i3 =\
               n2+get_global_id(1)+n*(n2-get_global_id(0)-1+n*get_global_id(2));
            float2 temp = a[i0];
            a[i0] = a[i3];
            a[i3] = a[i2];
            a[i2] = a[i1];
            a[i1] = temp;
         }

         // Call with (n/2,n/2,n)
         __kernel void rot270ev( __global float2 *a, unsigned n) {
            int n2=n/2;
            int i0 =\
               n2+get_global_id(0)+n*(n2+get_global_id(1)+n*get_global_id(2));
            int i1 =\
               n2-get_global_id(1)-1+n*(n2+get_global_id(0)+n*get_global_id(2));
            int i2 =\
               n2-get_global_id(0)-1\
               +n*(n2-get_global_id(1)-1+n*get_global_id(2));
            int i3 =\
               n2+get_global_id(1)\
               +n*(n2-get_global_id(0)-1+n*get_global_id(2));
            float2 temp = a[i0];
            a[i0] = a[i1];
            a[i1] = a[i2];
            a[i2] = a[i3];
            a[i3] = temp;
         }

         // Call with (n/2,n/2,n)
         __kernel void rot180ev( __global float2 *a, unsigned n) {
            //This could be made more efficient with (n,n/2,n) and do just
            //one interchange per item.
            int n2=n/2;
            int i0 =\
               n2+get_global_id(0)+n*(n2+get_global_id(1)+n*get_global_id(2));
            int i1 =\
               n2-get_global_id(1)-1+n*(n2+get_global_id(0)+n*get_global_id(2));
            int i2 =\
               n2-get_global_id(0)-1\
               +n*(n2-get_global_id(1)-1+n*get_global_id(2));
            int i3 =\
               n2+get_global_id(1)+n*(n2-get_global_id(0)-1+n*get_global_id(2));
            float2 temp = a[i0];
            a[i0] = a[i2];
            a[i2] = temp;
            temp = a[i1];
            a[i1] = a[i3];
            a[i3] = temp;
         }

         // Call with ((n+1)/2,(n-1)/2,n)
         __kernel void rot90odd( __global float2 *a, unsigned n) {
            int n2=(n+1)/2;
            int i0 =\
               n2-1+get_global_id(0)\
               +n*(n2+get_global_id(1)+n*get_global_id(2));
            int i1 =\
               n2-2-get_global_id(1)\
               +n*(n2-1+get_global_id(0)+n*get_global_id(2));
            int i2 =\
               n2-1-get_global_id(0)\
               +n*(n2-2-get_global_id(1)+n*get_global_id(2));
            int i3 =\
               n2+get_global_id(1)+n*(n2-1-get_global_id(0)+n*get_global_id(2));
            float2 temp = a[i0];
            a[i0] = a[i3];
            a[i3] = a[i2];
            a[i2] = a[i1];
            a[i1] = temp;
         }

         // Call with ((n+1)/2,(n-1)/2,n)
         __kernel void rot270odd( __global float2 *a, unsigned n) {
            int n2=(n+1)/2;
            int i0 =\
               n2-1+get_global_id(0)\
               +n*(n2+get_global_id(1)+n*get_global_id(2));
            int i1 =\
               n2-2-get_global_id(1)\
               +n*(n2-1+get_global_id(0)+n*get_global_id(2));
            int i2 =\
               n2-1-get_global_id(0)\
               +n*(n2-2-get_global_id(1)+n*get_global_id(2));
            int i3 =\
               n2+get_global_id(1)\
               +n*(n2-1-get_global_id(0)+n*get_global_id(2));
            float2 temp = a[i0];
            a[i0] = a[i1];
            a[i1] = a[i2];
            a[i2] = a[i3];
            a[i3] = temp;
         }

         // Call with ((n+1)/2,(n-1)/2,n)
         __kernel void rot180odd( __global float2 *a, unsigned n) {
            int n2=(n+1)/2;
            int i0 =\
               n2-1+get_global_id(0)\
               +n*(n2+get_global_id(1)+n*get_global_id(2));
            int i1 =\
               n2-2-get_global_id(1)\
               +n*(n2-1+get_global_id(0)+n*get_global_id(2));
            int i2 =\
               n2-1-get_global_id(0)
               +n*(n2-2-get_global_id(1)+n*get_global_id(2));
            int i3 =\
               n2+get_global_id(1)\
               +n*(n2-1-get_global_id(0)+n*get_global_id(2));
            float2 temp = a[i0];
            a[i0] = a[i2];
            a[i2] = temp;
            temp = a[i1];
            a[i1] = a[i3];
            a[i3] = temp;
         }

         """)

      if self.dev_dtype == np.complex128:
         self.prg = prg_double.build()
      else:
         self.prg = prg_single.build()

   @property
   def f(self):
      return self.f_dev.get().astype(np.complex128)

   @f.setter
   def f(self,f):
      self._checkf(f)
      cl.enqueue_copy(self.q,self.f_dev.data,
         f.astype(self.dev_dtype,order='C'))

   def rotation(self,R):
      euler = R.as_euler('xyx')
      ang=-euler[0]
      n90 = np.rint(ang*2.0/np.pi)
      dang = ang-n90*np.pi*0.5
      self.n90x1 = int(n90 % 4)
      scale0  = -np.tan(0.5*dang)
      x0, k0, x3 = self._getmultipliers(scale0)
      scale1 = np.sin(dang)
      x1, k1, x2 = self._getmultipliers(scale1)
      x1 = np.transpose(x1)*x3
      x2 = np.transpose(x2)*x0
      self.factors[0,:,:] = x0.astype(self.dev_dtype)
      self.factors[1,:,:] = k0.astype(self.dev_dtype)
      self.factors[2,:,:] = x1.astype(self.dev_dtype)
      self.factors[3,:,:] = k1.astype(self.dev_dtype)
      self.factors[4,:,:] = x2.astype(self.dev_dtype)
      self.factors[5,:,:] = x3.astype(self.dev_dtype)
      euler = R.as_euler('xyx')
      ang=euler[1]
      n90 = np.rint(ang*2.0/np.pi)
      dang = ang-n90*np.pi*0.5
      self.n90y = int(n90 % 4)
      scale0  = -np.tan(0.5*dang)
      x0, k0, x3 = self._getmultipliers(scale0)
      scale1 = np.sin(dang)
      x1, k1, x2 = self._getmultipliers(scale1)
      x1 = np.transpose(x1)*x3
      x2 = np.transpose(x2)*x0
      self.factors[6,:,:] = x0.astype(self.dev_dtype)
      self.factors[7,:,:] = k0.astype(self.dev_dtype)
      self.factors[8,:,:] = x1.astype(self.dev_dtype)
      self.factors[9,:,:] = k1.astype(self.dev_dtype)
      self.factors[10,:,:] = x2.astype(self.dev_dtype)
      self.factors[11,:,:] = x3.astype(self.dev_dtype)
      ang=-euler[2]
      n90 = np.rint(ang*2.0/np.pi)
      dang = ang-n90*np.pi*0.5
      self.n90x2 = int(n90 % 4)
      scale0  = -np.tan(0.5*dang)
      x0, k0, x3 = self._getmultipliers(scale0)
      scale1 = np.sin(dang)
      x1, k1, x2 = self._getmultipliers(scale1)
      x1 = np.transpose(x1)*x3
      x2 = np.transpose(x2)*x0
      self.factors[12,:,:] = x0.astype(self.dev_dtype)
      self.factors[13,:,:] = k0.astype(self.dev_dtype)
      self.factors[14,:,:] = x1.astype(self.dev_dtype)
      self.factors[15,:,:] = k1.astype(self.dev_dtype)
      self.factors[16,:,:] = x2.astype(self.dev_dtype)
      self.factors[17,:,:] = x3.astype(self.dev_dtype)
      cl.enqueue_copy(self.q,self.factors_dev.data,self.factors)
      if self.N%2 == 0:
         if self.n90x1 < 2:
            if self.n90x1 > 0:
               self.prg.rot90ev(self.q,(int(self.N/2),int(self.N/2),self.N),
                  None,self.f_dev.data,np.uint32(self.N))
         else:
            if self.n90x1 < 3:
               self.prg.rot180ev(self.q,(int(self.N/2),int(self.N/2),self.N),
                  None,self.f_dev.data,np.uint32(self.N))
            else:
               self.prg.rot270ev(self.q,(int(self.N/2),int(self.N/2),self.N),
                  None,self.f_dev.data,np.uint32(self.N))
      else:     
         if self.n90x1 < 2:
            if self.n90x1 > 0:
               self.prg.rot90odd(self.q,(int((self.N+1)/2),
                  int((self.N-1)/2),self.N),None,self.f_dev.data,
                  np.uint32(self.N))
         else:
            if self.n90x1 < 3:
               self.prg.rot180odd(self.q,(int((self.N+1)/2),
                  int((self.N-1)/2),self.N),None,self.f_dev.data,
                  np.uint32(self.N))
            else:
               self.prg.rot270odd(self.q,(int((self.N+1)/2),
                  int((self.N-1)/2),self.N),None,self.f_dev.data,
                  np.uint32(self.N))
      self.prg.multiply_ith(self.q,(self.N,self.N,self.N),None,
         self.factors_dev.data,self.f_dev.data,np.uint32(self.N),np.uint32(0))
      self.app.fft(self.f_dev)
      self.prg.multiply_ith(self.q,(self.N,self.N,self.N),None,
         self.factors_dev.data,self.f_dev.data,np.uint32(self.N),np.uint32(1))
      self.app.ifft(self.f_dev)
      self.prg.multiply_ith(self.q,(self.N,self.N,self.N),None,
         self.factors_dev.data,self.f_dev.data,np.uint32(self.N),np.uint32(2))
      self.prg.transposeyz(self.q,(self.N,self.N,self.N),None,
         self.f_dev.data,np.uint32(self.N))
      self.app.fft(self.f_dev)
      self.prg.multiply_ith(self.q,(self.N,self.N,self.N),None,
         self.factors_dev.data,self.f_dev.data,np.uint32(self.N),np.uint32(3))
      self.app.ifft(self.f_dev)
      self.prg.transposeyz(self.q,(self.N,self.N,self.N),None,
         self.f_dev.data,np.uint32(self.N))
      self.prg.multiply_ith(self.q,(self.N,self.N,self.N),None,
         self.factors_dev.data,self.f_dev.data,np.uint32(self.N),np.uint32(4))
      self.app.fft(self.f_dev)
      self.prg.multiply_ith(self.q,(self.N,self.N,self.N),None,
         self.factors_dev.data,self.f_dev.data,np.uint32(self.N),np.uint32(1))
      self.app.ifft(self.f_dev)
      self.prg.multiply_ith(self.q,(self.N,self.N,self.N),None,
         self.factors_dev.data,self.f_dev.data,np.uint32(self.N),np.uint32(5))

      self.prg.transposexy(self.q,(self.N,self.N,self.N),None,
         self.f_dev.data,np.uint32(self.N))
      if self.N%2 == 0:
         if self.n90y < 2:
            if self.n90y > 0:
               self.prg.rot90ev(self.q,(int(self.N/2),int(self.N/2),self.N),
                  None,self.f_dev.data,np.uint32(self.N))
         else:
            if self.n90y < 3:
               self.prg.rot180ev(self.q,(int(self.N/2),int(self.N/2),self.N),
                  None,self.f_dev.data,np.uint32(self.N))
            else:
               self.prg.rot270ev(self.q,(int(self.N/2),int(self.N/2),self.N),
                  None,self.f_dev.data,np.uint32(self.N))
      else:     
         if self.n90y < 2:
            if self.n90y > 0:
               self.prg.rot90odd(self.q,(int((self.N+1)/2),
                  int((self.N-1)/2),self.N),None,self.f_dev.data,
                  np.uint32(self.N))
         else:
            if self.n90y < 3:
               self.prg.rot180odd(self.q,(int((self.N+1)/2),
                  int((self.N-1)/2),self.N),None,self.f_dev.data,
                  np.uint32(self.N))
            else:
               self.prg.rot270odd(self.q,(int((self.N+1)/2),
                  int((self.N-1)/2),self.N),None,self.f_dev.data,
                  np.uint32(self.N))
      self.prg.multiply_ith(self.q,(self.N,self.N,self.N),None,
         self.factors_dev.data,self.f_dev.data,np.uint32(self.N),np.uint32(6))
      self.app.fft(self.f_dev)
      self.prg.multiply_ith(self.q,(self.N,self.N,self.N),None,
         self.factors_dev.data,self.f_dev.data,np.uint32(self.N),np.uint32(7))
      self.app.ifft(self.f_dev)
      self.prg.multiply_ith(self.q,(self.N,self.N,self.N),None,
         self.factors_dev.data,self.f_dev.data,np.uint32(self.N),np.uint32(8))
      self.prg.transposeyz(self.q,(self.N,self.N,self.N),None,
         self.f_dev.data,np.uint32(self.N))
      self.app.fft(self.f_dev)
      self.prg.multiply_ith(self.q,(self.N,self.N,self.N),None,
         self.factors_dev.data,self.f_dev.data,np.uint32(self.N),np.uint32(9))
      self.app.ifft(self.f_dev)
      self.prg.transposeyz(self.q,(self.N,self.N,self.N),None,
         self.f_dev.data,np.uint32(self.N))
      self.prg.multiply_ith(self.q,(self.N,self.N,self.N),None,
         self.factors_dev.data,self.f_dev.data,np.uint32(self.N),np.uint32(10))
      self.app.fft(self.f_dev)
      self.prg.multiply_ith(self.q,(self.N,self.N,self.N),None,
         self.factors_dev.data,self.f_dev.data,np.uint32(self.N),np.uint32(7))
      self.app.ifft(self.f_dev)
      self.prg.multiply_ith(self.q,(self.N,self.N,self.N),None,
         self.factors_dev.data,self.f_dev.data,np.uint32(self.N),np.uint32(11))
      self.prg.transposexy(self.q,(self.N,self.N,self.N),None,
         self.f_dev.data,np.uint32(self.N))

      if self.N%2 == 0:
         if self.n90x2 < 2:
            if self.n90x2 > 0:
               self.prg.rot90ev(self.q,(int(self.N/2),int(self.N/2),self.N),
                  None,self.f_dev.data,np.uint32(self.N))
         else:
            if self.n90x2 < 3:
               self.prg.rot180ev(self.q,(int(self.N/2),int(self.N/2),self.N),
                  None,self.f_dev.data,np.uint32(self.N))
            else:
               self.prg.rot270ev(self.q,(int(self.N/2),int(self.N/2),self.N),
                  None,self.f_dev.data,np.uint32(self.N))
      else:     
         if self.n90x2 < 2:
            if self.n90x2 > 0:
               self.prg.rot90odd(self.q,(int((self.N+1)/2),
                  int((self.N-1)/2),self.N),None,self.f_dev.data,
                  np.uint32(self.N))
         else:
            if self.n90x2 < 3:
               self.prg.rot180odd(self.q,(int((self.N+1)/2),
                  int((self.N-1)/2),self.N),None,self.f_dev.data,
                  np.uint32(self.N))
            else:
               self.prg.rot270odd(self.q,(int((self.N+1)/2),
                  int((self.N-1)/2),self.N),None,self.f_dev.data,
                  np.uint32(self.N))
      self.prg.multiply_ith(self.q,(self.N,self.N,self.N),None,
         self.factors_dev.data,self.f_dev.data,np.uint32(self.N),np.uint32(12))
      self.app.fft(self.f_dev)
      self.prg.multiply_ith(self.q,(self.N,self.N,self.N),None,
         self.factors_dev.data,self.f_dev.data,np.uint32(self.N),np.uint32(13))
      self.app.ifft(self.f_dev)
      self.prg.multiply_ith(self.q,(self.N,self.N,self.N),None,
         self.factors_dev.data,self.f_dev.data,np.uint32(self.N),np.uint32(14))
      self.prg.transposeyz(self.q,(self.N,self.N,self.N),None,
         self.f_dev.data,np.uint32(self.N))
      self.app.fft(self.f_dev)
      self.prg.multiply_ith(self.q,(self.N,self.N,self.N),None,
         self.factors_dev.data,self.f_dev.data,np.uint32(self.N),np.uint32(15))
      self.app.ifft(self.f_dev)
      self.prg.transposeyz(self.q,(self.N,self.N,self.N),None,
         self.f_dev.data,np.uint32(self.N))
      self.prg.multiply_ith(self.q,(self.N,self.N,self.N),None,
         self.factors_dev.data,self.f_dev.data,np.uint32(self.N),np.uint32(16))
      self.app.fft(self.f_dev)
      self.prg.multiply_ith(self.q,(self.N,self.N,self.N),None,
         self.factors_dev.data,self.f_dev.data,np.uint32(self.N),np.uint32(13))
      self.app.ifft(self.f_dev)
      self.prg.multiply_ith(self.q,(self.N,self.N,self.N),None,
         self.factors_dev.data,self.f_dev.data,np.uint32(self.N),np.uint32(17))

   def _rotate3Dx(self,angin):
      pass

   def _rotate3Dy(self,ang):
      pass
