import numpy as np
import scipy
import scipy.fft as fft

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

        keep_last_even_k : default False. The last k for even N has an ambiguous
        sign. Keeping one sign only, makes real data become complex, so
        this is always False for real data.
        

    Methods:
      rotation(R): R is a rotation specified as a
                   scipy.spatial.transform.Rotation 
   """

   def __init__(self,f3d,keep_last_even_k=False):
      self.N = 0
      self.f = f3d
      self.keep_last_even_k = keep_last_even_k
      self.dtf = self._f.dtype
      self._setkorderc0()

   @property
   def f(self):
      return self._f

   @f.setter
   def f(self,f):
      self._checkf(f)
      self._f = f.copy()

   def _checkf(self,f3d):
#check f3d is 3d, cubic
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

   def rotation(self,R):
      euler = R.as_euler('xyx')
      if self.dtf == np.float64 or self.dtf == np.float32:
         self._rotate3Dxr(euler[0])
         self._rotate3Dyr(euler[1])
         self._rotate3Dxr(euler[2])
      else:
         self._rotate3Dx(euler[0])
         self._rotate3Dy(euler[1])
         self._rotate3Dx(euler[2])

   def _rotate3Dxr(self,angin):
# angle negative since done in order z then y
      ang = -angin
      n90 = np.rint(ang*2.0/np.pi)
      dang = ang-n90*np.pi*0.5
      n90 = int(n90 % 4)
      scale0  = -np.tan(0.5*dang)
      k0 = self._getkmult(scale0)
      k0 = np.transpose(k0).copy()
      scale1 = np.sin(dang)
      k1 = self._getkmult(scale1)
      n2 = self.N//2+1
      for i in range(self.N):
         ftmp = self._f[i,:,:]
         if (n90 == 1):
            ftmp = np.rot90(ftmp,1,axes=(1,0))
         if (n90 == 2):
            ftmp = np.rot90(ftmp,2,axes=(1,0))
         if (n90 == 3):
            ftmp = np.rot90(ftmp,-1,axes=(1,0))
         ftmp = fft.irfft(fft.rfft(ftmp,axis=1)*k0[:,0:n2],self.N,axis=1)
         ftmp = fft.irfft(fft.rfft(ftmp,axis=0)*k1[0:n2,:],self.N,axis=0)
         ftmp = fft.irfft(fft.rfft(ftmp,axis=1)*k0[:,0:n2],self.N,axis=1)
         self._f[i,:,:] = ftmp

   def _rotate3Dyr(self,ang):
# identical to x rotation except angle positive since
# this is done in the order z then x, and the array slices
# in for loop are x-z.
      n90 = np.rint(ang*2.0/np.pi)
      dang = ang-n90*np.pi*0.5
      n90 = int(n90 % 4)
      scale0  = -np.tan(0.5*dang)
      k0 = self._getkmult(scale0)
      k0 = np.transpose(k0).copy()
      scale1 = np.sin(dang)
      k1 = self._getkmult(scale1)
      n2 = self.N//2+1
      for i in range(self.N):
         ftmp = self._f[:,i,:]
         if (n90 == 1):
            ftmp = np.rot90(ftmp,1,axes=(1,0))
         if (n90 == 2):
            ftmp = np.rot90(ftmp,2,axes=(1,0))
         if (n90 == 3):
            ftmp = np.rot90(ftmp,-1,axes=(1,0))
         ftmp = fft.irfft(fft.rfft(ftmp,axis=1)*k0[:,0:n2],self.N,axis=1)
         ftmp = fft.irfft(fft.rfft(ftmp,axis=0)*k1[0:n2,:],self.N,axis=0)
         ftmp = fft.irfft(fft.rfft(ftmp,axis=1)*k0[:,0:n2],self.N,axis=1)
         self._f[:,i,:] = ftmp

   def _rotate3Dx(self,angin):
# angle negative since done in order z then y
      ang = -angin
      n90 = np.rint(ang*2.0/np.pi)
      dang = ang-n90*np.pi*0.5
      n90 = int(n90 % 4)
      scale0  = -np.tan(0.5*dang)
      k0 = self._getkmult(scale0)
      k0 = np.transpose(k0).copy()
      scale1 = np.sin(dang)
      k1 = self._getkmult(scale1)
      for i in range(self.N):
         ftmp = self._f[i,:,:]
         if (n90 == 1):
            ftmp = np.rot90(ftmp,1,axes=(1,0))
         if (n90 == 2):
            ftmp = np.rot90(ftmp,2,axes=(1,0))
         if (n90 == 3):
            ftmp = np.rot90(ftmp,-1,axes=(1,0))
         ftmp = fft.ifft(fft.fft(ftmp,axis=1)*k0,axis=1)
         ftmp = fft.ifft(fft.fft(ftmp,axis=0)*k1,axis=0)
         ftmp = fft.ifft(fft.fft(ftmp,axis=1)*k0,axis=1)
         self._f[i,:,:] = ftmp

   def _rotate3Dy(self,ang):
# identical to x rotation except angle positive since
# this is done in the order z then x, and the array slices
# in for loop are x-z.
      n90 = np.rint(ang*2.0/np.pi)
      dang = ang-n90*np.pi*0.5
      n90 = int(n90 % 4)
      scale0  = -np.tan(0.5*dang)
      k0 = self._getkmult(scale0)
      k0 = np.transpose(k0).copy()
      scale1 = np.sin(dang)
      k1 = self._getkmult(scale1)
      for i in range(self.N):
         ftmp = self._f[:,i,:]
         if (n90 == 1):
            ftmp = np.rot90(ftmp,1,axes=(1,0))
         if (n90 == 2):
            ftmp = np.rot90(ftmp,2,axes=(1,0))
         if (n90 == 3):
            ftmp = np.rot90(ftmp,-1,axes=(1,0))
         ftmp = fft.ifft(fft.fft(ftmp,axis=1)*k0,axis=1)
         ftmp = fft.ifft(fft.fft(ftmp,axis=0)*k1,axis=0)
         ftmp = fft.ifft(fft.fft(ftmp,axis=1)*k0,axis=1)
         self._f[:,i,:] = ftmp

   def _setkorderc0(self):
      self.c0 = 0.5*(self.N-1)
      if self.N % 2 == 0:
         kint0 = np.arange(-self.N/2,self.N/2)
      else:
         kint0 = np.arange((1-self.N)/2,(1+self.N)/2)
      self.kint = np.zeros(self.N,np.float64)
      self.kint[(self.N+1)//2:] = kint0[0:self.N//2]
      self.kint[0:(self.N+1)//2] = kint0[self.N//2:]

   def _getkmult(self,scale):
      nint = np.arange(self.N)
      k0 = np.zeros((self.N,self.N),np.complex128)
      ck = -1j*2.0*np.pi/self.N*scale
      for k in range(self.N):
         k0[k,:] = np.exp(ck*(nint-self.c0)*self.kint[k])
      if self.N % 2 == 0 and not self.keep_last_even_k:
         k0[self.N//2] = 0.0
      if self.dtf == np.complex64 or self.dtf == np.float32:
         k0 = k0.astype(dtype=np.complex64)
      return k0

class rotate3Dlegacy(rotate3D):
   r"""
    This is identical to rotate3D except that the shear orders are
    the same as in Joe's original code, and the last lastk value is not
    zeroed. It is somewhat less efficient since the arrays are transposed
    and then transposed back.
  """

   def __init__(self,f3d):
      super().__init__(f3d,keep_last_even_k=True)

   def rotation(self,R):
      euler = R.as_euler('xyx')
      self._f = np.transpose(self._f,axes=(0,2,1))
      self._rotate3Dx(-euler[0])
      self._f = np.transpose(self._f,axes=(1,2,0))
      self._rotate3Dy(-euler[1])
      self._f = np.transpose(self._f,axes=(2,0,1))
      self._rotate3Dx(-euler[2])
      self._f = np.transpose(self._f,axes=(0,2,1))

try:
   import pyopencl as cl
   import pyopencl.array
   import pyvkfft.opencl
   import reborn
   import reborn.simulate
   import reborn.simulate.clcore
except:
   import reborn
   reborn.utils.warn("pyvkfft or its dependencies could not be set up correcty")
   reborn.utils.warn("rotate3Dvkfft should not be used")

class rotate3Dvkfft(rotate3D):
   r"""
    This should give results identical to rotate3D for sizes
    that are products of powers of 2,3,5,7,11,13.  It uses the pyvkfft
    wrapper to VkFFT to perform Fourier transforms on a gpu. Since this
    is my first opencl code, it is no doubt written inefficiently.
   """

   def __init__(self,f3d,keep_last_even_k=False):
      self.keep_last_even_k = keep_last_even_k
      self.N = 0
      self.f_dev = None
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
      self.dtf = f3d.dtype
      self.ctx = reborn.simulate.clcore.create_some_gpu_context()
      self.q = cl.CommandQueue(self.ctx)
      self.rotsize = ((self.N+1)//2,self.N//2, self.N)
      self.transize = (self.N,self.N,self.N)
      if self.dtf == np.float64 or self.dtf == np.float32:
         if self.dtf == np.float64:
            self.app = pyvkfft.opencl.VkFFTApp((self.N,self.N,self.N+2),
               dtype=np.complex128,queue=self.q,ndim=1,r2c=True)
         else:
            self.app = pyvkfft.opencl.VkFFTApp((self.N,self.N,self.N+2),
               dtype=np.complex64,queue=self.q,ndim=1,r2c=True)
         ftmp = np.ndarray((self.N,self.N,2*(self.N//2+1)),self.dtf)
         ftmp[:,:,0:self.N] = f3d
         self.f_dev = cl.array.to_device(self.q,ftmp)
         ftmp = None
         self.multsize = (self.N//2+1,self.N,self.N)
      else:
         self.app = pyvkfft.opencl.VkFFTApp((self.N,self.N,self.N),
            dtype=self.dtf,queue=self.q,ndim=1)
         self.f_dev = cl.array.to_device(self.q,f3d)
         self.multsize = (self.N,self.N,self.N)
      if self.dtf == np.float64:
         self.dtfac = np.complex128
      elif self.dtf == np.float32:
         self.dtfac = np.complex64
      else:
         self.dtfac = self.dtf
      self.factors = np.ndarray((6,self.N,self.N),self.dtfac)
      self.factors_dev = cl.array.to_device(self.q,self.factors)
      self._setkorderc0()
      #stupid routines on gpu -- improve me
      src_double = """
#ifdef R2C
         __kernel void transposeyz(  __global double *a, unsigned n) {
            double temp;
            int n2 = 2*(n/2+1);
            int ii = get_global_id(0)+\
               (get_global_id(1)+get_global_id(2)*n)*(n2);
            int io = get_global_id(1)+\
               (get_global_id(0)+get_global_id(2)*n)*(n2);
#else
         __kernel void transposeyz(  __global double2 *a, unsigned n) {
            double2 temp;
            int ii = get_global_id(0)+(get_global_id(1)+get_global_id(2)*n)*n;
            int io = get_global_id(1)+(get_global_id(0)+get_global_id(2)*n)*n;
#endif
            if (ii <= io) {
               temp = a[io];
               a[io] = a[ii];
               a[ii] = temp;
            }
         }
   
#ifdef R2C
         __kernel void transposexy( __global double *a, unsigned n) {
            double temp;
            int n2 = 2*(n/2+1);
            int ii = get_global_id(0)+\
               (get_global_id(1)+get_global_id(2)*n)*(n2);
            int io = get_global_id(0)+\
               (get_global_id(2)+get_global_id(1)*n)*(n2);
#else
         __kernel void transposexy( __global double2 *a, unsigned n) {
            double2 temp;
            int ii = get_global_id(0)+(get_global_id(1)+get_global_id(2)*n)*n;
            int io = get_global_id(0)+(get_global_id(2)+get_global_id(1)*n)*n;
#endif
            if (ii <= io) {
               temp = a[io];
               a[io] = a[ii];
               a[ii] = temp;
            }
         }
   
         __kernel void multiply_ith( __global double2 *factor,\
             __global double2 *a, unsigned n, unsigned ith) {
            int ifac = get_global_id(0)+n*(get_global_id(1)+n*ith);
#ifdef R2C
            int n2 = n/2+1;
            int i = get_global_id(0)+\
               (get_global_id(1)+get_global_id(2)*n)*n2;
#else
            int i = get_global_id(0)+(get_global_id(1)+get_global_id(2)*n)*n;
#endif
            double tempr = a[i].x;
            a[i].x = factor[ifac].x*a[i].x-factor[ifac].y*a[i].y;
            a[i].y = factor[ifac].x*a[i].y+factor[ifac].y*tempr;
         }

#ifdef EVEN_N
         // Even n call with (n/2,n/2,n)
#ifdef R2C
         __kernel void rot90( __global double *a, unsigned n) {
            double temp;
            int n2=n/2;
            int i0 = n2+get_global_id(0)\
               +(n+2)*(n2+get_global_id(1)+n*get_global_id(2));
            int i1 = n2-get_global_id(1)-1\
               +(n+2)*(n2+get_global_id(0)+n*get_global_id(2));
            int i2 =n2-get_global_id(0)-1\
               +(n+2)*(n2-get_global_id(1)-1+n*get_global_id(2));
            int i3 =n2+get_global_id(1)
               +(n+2)*(n2-get_global_id(0)-1+n*get_global_id(2));
#else
         __kernel void rot90( __global double2 *a, unsigned n) {
            double2 temp;
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
#endif
            temp = a[i0];
            a[i0] = a[i3];
            a[i3] = a[i2];
            a[i2] = a[i1];
            a[i1] = temp;
         }

         // Even n call with (n/2,n/2,n)
#ifdef R2C
         __kernel void rot270( __global double *a, unsigned n) {
            double temp;
            int n2=n/2;
            int i0 = n2+get_global_id(0)\
               +(n+2)*(n2+get_global_id(1)+n*get_global_id(2));
            int i1 = n2-get_global_id(1)-1\
               +(n+2)*(n2+get_global_id(0)+n*get_global_id(2));
            int i2 = n2-get_global_id(0)-1\
               +(n+2)*(n2-get_global_id(1)-1+n*get_global_id(2));
            int i3 = n2+get_global_id(1)\
               +(n+2)*(n2-get_global_id(0)-1+n*get_global_id(2));
#else
         __kernel void rot270( __global double2 *a, unsigned n) {
            double2 temp;
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
#endif
            temp = a[i0];
            a[i0] = a[i1];
            a[i1] = a[i2];
            a[i2] = a[i3];
            a[i3] = temp;
         }

         //This could be made more efficient with (n,n/2,n) and do just
         //one interchange per item.
         // Even n call with (n/2,n/2,n)
#ifdef R2C
         __kernel void rot180( __global double *a, unsigned n) {
            double temp;
            int n2=n/2;
            int i0 = n2+get_global_id(0)\
               +(n+2)*(n2+get_global_id(1)+n*get_global_id(2));
            int i1 = n2-get_global_id(1)-1\
               +(n+2)*(n2+get_global_id(0)+n*get_global_id(2));
            int i2 = n2-get_global_id(0)-1\
               +(n+2)*(n2-get_global_id(1)-1+n*get_global_id(2));
            int i3 = n2+get_global_id(1)\
               +(n+2)*(n2-get_global_id(0)-1+n*get_global_id(2));
#else
         __kernel void rot180( __global double2 *a, unsigned n) {
            double2 temp;
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
#endif
            temp = a[i0];
            a[i0] = a[i2];
            a[i2] = temp;
            temp = a[i1];
            a[i1] = a[i3];
            a[i3] = temp;
         }
#else
         // Odd n, call with ((n+1)/2,(n-1)/2,n)
#ifdef R2C
         __kernel void rot90( __global double *a, unsigned n) {
            double temp;
            int n2=(n+1)/2;
            int i0 = n2-1+get_global_id(0)\
               +(n+1)*(n2+get_global_id(1)+n*get_global_id(2));
            int i1 = n2-2-get_global_id(1)\
               +(n+1)*(n2-1+get_global_id(0)+n*get_global_id(2));
            int i2 = n2-1-get_global_id(0)\
               +(n+1)*(n2-2-get_global_id(1)+n*get_global_id(2));
            int i3 = n2+get_global_id(1)\
               +(n+1)*(n2-1-get_global_id(0)+n*get_global_id(2));
#else
         __kernel void rot90( __global double2 *a, unsigned n) {
            double2 temp;
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
#endif
            temp = a[i0];
            a[i0] = a[i3];
            a[i3] = a[i2];
            a[i2] = a[i1];
            a[i1] = temp;
         }

         // Odd n, call with ((n+1)/2,(n-1)/2,n)
#ifdef R2C
         __kernel void rot270( __global double *a, unsigned n) {
            double temp;
            int n2=(n+1)/2;
            int i0 =\
               n2-1+get_global_id(0)\
               +(n+1)*(n2+get_global_id(1)+n*get_global_id(2));
            int i1 =\
               n2-2-get_global_id(1)\
               +(n+1)*(n2-1+get_global_id(0)+n*get_global_id(2));
            int i2 =\
               n2-1-get_global_id(0)\
               +(n+1)*(n2-2-get_global_id(1)+n*get_global_id(2));
            int i3 =\
               n2+get_global_id(1)\
               +(n+1)*(n2-1-get_global_id(0)+n*get_global_id(2));
#else
         __kernel void rot270( __global double2 *a, unsigned n) {
            double2 temp;
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
#endif
            temp = a[i0];
            a[i0] = a[i1];
            a[i1] = a[i2];
            a[i2] = a[i3];
            a[i3] = temp;
         }

         // Odd n, call with ((n+1)/2,(n-1)/2,n)
#ifdef R2C
         __kernel void rot180( __global double *a, unsigned n) {
            double temp;
            int n2=(n+1)/2;
            int i0 =\
               n2-1+get_global_id(0)\
               +(n+1)*(n2+get_global_id(1)+n*get_global_id(2));
            int i1 =\
               n2-2-get_global_id(1)\
               +(n+1)*(n2-1+get_global_id(0)+n*get_global_id(2));
            int i2 =\
               n2-1-get_global_id(0)
               +(n+1)*(n2-2-get_global_id(1)+n*get_global_id(2));
            int i3 =\
               n2+get_global_id(1)\
               +(n+1)*(n2-1-get_global_id(0)+n*get_global_id(2));
#else
         __kernel void rot180( __global double2 *a, unsigned n) {
            double2 temp;
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
#endif
            temp = a[i0];
            a[i0] = a[i2];
            a[i2] = temp;
            temp = a[i1];
            a[i1] = a[i3];
            a[i3] = temp;
         }
#endif
         """
      opts = ""
      if self.N%2 == 0:
         opts += " -DEVEN_N "

      if self.dtf == np.float64 or self.dtf == np.float32:
         opts += " -DR2C "

      if self.dtf == np.complex128 or self.dtf == np.float64:
         self.prg = cl.Program(self.ctx,src_double).build(options=opts)
      else:
         # Make sure no variables or routine names contain the string
         # double or this simple substitution will not work.
         self.prg = cl.Program(self.ctx,
            src_double.replace("double","float")).build(options=opts)

   @property
   def f(self):
      if self.dtf == np.float64 or self.dtf == np.float32:
         return (self.f_dev.get()[:,:,0:self.N]).astype(self.dtf)
      else:
         return self.f_dev.get().astype(self.dtf)

   @f.setter
   def f(self,f):
      self._checkf(f)
      if self.dtf == np.float64 or self.dtf == np.float32:
         ftmp = np.ndarray((self.N,self.N,2*(self.N//2+1)),dtype=self.dtf)
         ftmp[:,:,0:self.N] = f
         cl.enqueue_copy(self.q,self.f_dev.data,
            ftmp.astype(self.dtf,order='C'))
         ftmp = None
      else:
         cl.enqueue_copy(self.q,self.f_dev.data,
            f.astype(self.dtf,order='C'))

   def rotation(self,R):
      euler = R.as_euler('xyx')
      ang=-euler[0]
      n90 = np.rint(ang*2.0/np.pi)
      dang = ang-n90*np.pi*0.5
      self.n90x1 = int(n90 % 4)
      scale0  = -np.tan(0.5*dang)
      k0 = self._getkmult(scale0)
      k0 = np.transpose(k0).copy()
      scale1 = np.sin(dang)
      k1 = self._getkmult(scale1)
      k1 = np.transpose(k1).copy()
      self.factors[0,:,:] = k0.astype(self.dtfac)
      self.factors[1,:,:] = k1.astype(self.dtfac)
      ang=euler[1]
      n90 = np.rint(ang*2.0/np.pi)
      dang = ang-n90*np.pi*0.5
      self.n90y = int(n90 % 4)
      scale0  = -np.tan(0.5*dang)
      k0 = self._getkmult(scale0)
      k0 = np.transpose(k0).copy()
      scale1 = np.sin(dang)
      k1 = self._getkmult(scale1)
      k1 = np.transpose(k1).copy()
      self.factors[2,:,:] = k0.astype(self.dtfac)
      self.factors[3,:,:] = k1.astype(self.dtfac)
      ang=-euler[2]
      n90 = np.rint(ang*2.0/np.pi)
      dang = ang-n90*np.pi*0.5
      self.n90x2 = int(n90 % 4)
      scale0  = -np.tan(0.5*dang)
      k0 = self._getkmult(scale0)
      k0 = np.transpose(k0).copy()
      scale1 = np.sin(dang)
      k1 = self._getkmult(scale1)
      k1 = np.transpose(k1).copy()
      self.factors[4,:,:] = k0.astype(self.dtfac)
      self.factors[5,:,:] = k1.astype(self.dtfac)
      cl.enqueue_copy(self.q,self.factors_dev.data,self.factors)
      if self.n90x1 < 2:
         if self.n90x1 > 0:
            self.prg.rot90(self.q,self.rotsize,
               None,self.f_dev.data,np.uint32(self.N))
      else:
         if self.n90x1 < 3:
            self.prg.rot180(self.q,self.rotsize,
               None,self.f_dev.data,np.uint32(self.N))
         else:
            self.prg.rot270(self.q,self.rotsize,
               None,self.f_dev.data,np.uint32(self.N))
      self.app.fft(self.f_dev)
      self.prg.multiply_ith(self.q,self.multsize,None,
         self.factors_dev.data,self.f_dev.data,np.uint32(self.N),
         np.uint32(0))
      self.app.ifft(self.f_dev)
      self.prg.transposeyz(self.q,self.transize,None,
         self.f_dev.data,np.uint32(self.N))
      self.app.fft(self.f_dev)
      self.prg.multiply_ith(self.q,self.multsize,None,
         self.factors_dev.data,self.f_dev.data,np.uint32(self.N),
         np.uint32(1))
      self.app.ifft(self.f_dev)
      self.prg.transposeyz(self.q,self.transize,None,
         self.f_dev.data,np.uint32(self.N))
      self.app.fft(self.f_dev)
      self.prg.multiply_ith(self.q,self.multsize,None,
         self.factors_dev.data,self.f_dev.data,np.uint32(self.N),
         np.uint32(0))
      self.app.ifft(self.f_dev)

      self.prg.transposexy(self.q,self.transize,None,
         self.f_dev.data,np.uint32(self.N))
      if self.n90y < 2:
         if self.n90y > 0:
            self.prg.rot90(self.q,self.rotsize,
               None,self.f_dev.data,np.uint32(self.N))
      else:
         if self.n90y < 3:
            self.prg.rot180(self.q,self.rotsize,
               None,self.f_dev.data,np.uint32(self.N))
         else:
            self.prg.rot270(self.q,self.rotsize,
               None,self.f_dev.data,np.uint32(self.N))
      self.app.fft(self.f_dev)
      self.prg.multiply_ith(self.q,self.multsize,None,
         self.factors_dev.data,self.f_dev.data,np.uint32(self.N),
         np.uint32(2))
      self.app.ifft(self.f_dev)
      self.prg.transposeyz(self.q,self.transize,None,
         self.f_dev.data,np.uint32(self.N))
      self.app.fft(self.f_dev)
      self.prg.multiply_ith(self.q,self.multsize,None,
         self.factors_dev.data,self.f_dev.data,np.uint32(self.N),
         np.uint32(3))
      self.app.ifft(self.f_dev)
      self.prg.transposeyz(self.q,self.transize,None,
         self.f_dev.data,np.uint32(self.N))
      self.app.fft(self.f_dev)
      self.prg.multiply_ith(self.q,self.multsize,None,
         self.factors_dev.data,self.f_dev.data,np.uint32(self.N),
         np.uint32(2))
      self.app.ifft(self.f_dev)
      self.prg.transposexy(self.q,self.transize,None,
         self.f_dev.data,np.uint32(self.N))

      if self.n90x2 < 2:
         if self.n90x2 > 0:
            self.prg.rot90(self.q,self.rotsize,
               None,self.f_dev.data,np.uint32(self.N))
      else:
         if self.n90x2 < 3:
            self.prg.rot180(self.q,self.rotsize,
               None,self.f_dev.data,np.uint32(self.N))
         else:
            self.prg.rot270(self.q,self.rotsize,
               None,self.f_dev.data,np.uint32(self.N))
      self.app.fft(self.f_dev)
      self.prg.multiply_ith(self.q,self.multsize,None,
         self.factors_dev.data,self.f_dev.data,np.uint32(self.N),
         np.uint32(4))
      self.app.ifft(self.f_dev)
      self.prg.transposeyz(self.q,self.transize,None,
         self.f_dev.data,np.uint32(self.N))
      self.app.fft(self.f_dev)
      self.prg.multiply_ith(self.q,self.multsize,None,
         self.factors_dev.data,self.f_dev.data,np.uint32(self.N),
         np.uint32(5))
      self.app.ifft(self.f_dev)
      self.prg.transposeyz(self.q,self.transize,None,
         self.f_dev.data,np.uint32(self.N))
      self.app.fft(self.f_dev)
      self.prg.multiply_ith(self.q,self.multsize,None,
         self.factors_dev.data,self.f_dev.data,np.uint32(self.N),
         np.uint32(4))
      self.app.ifft(self.f_dev)

   def _rotate3Dx(self,angin):
      pass

   def _rotate3Dy(self,ang):
      pass
