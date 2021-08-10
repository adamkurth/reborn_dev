import numpy as np
import scipy.fft as fft
import pyopencl as cl
import pyopencl.array
#import gpyfft
#import gpyfft.gpyfftlib
import time
import pyvkfft.opencl

# setup opencl
ctx = cl.create_some_context()
q = cl.CommandQueue(ctx)
mem_pool = cl.tools.MemoryPool(cl.tools.ImmediateAllocator(q))
#mem_pool = None #goes back to default allocator

#set up with random double complex arrays
dt = np.float64
dtc = np.complex128
Ns = (243, 256, 384, 500, 512)

for N in Ns:
   tmpr = np.random.rand(N*N).astype(dt)
   tmpi = np.random.rand(N*N).astype(dt)
   x0 = np.reshape(tmpr+1j*tmpi,(N,N))
   tmpr = np.random.rand(N*N).astype(dt)
   tmpi = np.random.rand(N*N).astype(dt)
   x1 = np.reshape(tmpr+1j*tmpi,(N,N))
   tmpr = np.random.rand(N*N).astype(dt)
   tmpi = np.random.rand(N*N).astype(dt)
   x2 = np.reshape(tmpr+1j*tmpi,(N,N))
   tmpr = np.random.rand(N*N).astype(dt)
   tmpi = np.random.rand(N*N).astype(dt)
   k0 = np.reshape(tmpr+1j*tmpi,(N,N))
   tmpr = np.random.rand(N*N).astype(dt)
   tmpi = np.random.rand(N*N).astype(dt)
   k1 = np.reshape(tmpr+1j*tmpi,(N,N))
   tmpr = np.random.rand(N*N).astype(dt)
   tmpi = np.random.rand(N*N).astype(dt)
   k1t = np.transpose(k1).copy()
   f = np.reshape(tmpr+1j*tmpi,(N,N))
   print(f.dtype)
   
# scipy fft test version
   t0 = time.time()
   ftmp1 = f*x0
   ftmp2 = fft.fft(ftmp1,axis=1)
   ftmp3 = ftmp2*k0
   ftmp4 = fft.ifft(ftmp3,axis=1)
   ftmp5 = ftmp4*x1
   ftmp6 = fft.fft(ftmp5,axis=0)
   ftmp7 = ftmp6*k1
   ftmp8 = fft.ifft(ftmp7,axis=0)
   ftmp9 = ftmp8*x2
   ftmp10 = fft.fft(ftmp9,axis=1)
   ftmp11 = ftmp10*k0
   ftmp12 = fft.ifft(ftmp11,axis=1)
   ftmp13 = ftmp12*x1
   t1 = time.time()
   print("Stupid scipy fft ",t1-t0," seconds")
   
# set up opencl arrays on device
   x0_dev = cl.array.to_device(q,x0,allocator=mem_pool)
   x1_dev = cl.array.to_device(q,x1,allocator=mem_pool)
   x2_dev = cl.array.to_device(q,x2,allocator=mem_pool)
   k0_dev = cl.array.to_device(q,k0,allocator=mem_pool)
   k1_dev = cl.array.to_device(q,k1,allocator=mem_pool)
   k1t_dev = cl.array.to_device(q,k1t,allocator=mem_pool)
   
   f_dev = cl.array.to_device(q,f,allocator=mem_pool)
   ft_dev = cl.array.to_device(q,f,allocator=mem_pool)
#   trans1 = gpyfft.fft.FFT(ctx,q,f_dev,axes=(1,))
#   trans1.enqueue_arrays(data=f_dev) #force execution to complete planning
   f_dev = cl.array.to_device(q,f,allocator=mem_pool)
#   trans2 = gpyfft.fft.FFT(ctx,q,f_dev,axes=(0,))
#   trans2.enqueue_arrays(data=f_dev) #force execution to complete planning
   #this execution doesn't seem to make any difference

#stupid complex transpose for pyvkfft on gpu
   prg = cl.Program(ctx, """
      __kernel void transpose( __global double *at, __global double *a,
         unsigned n) {
         int ii = 2*(get_global_id(0)+get_global_id(1)*n);
         int io = 2*(get_global_id(1)+get_global_id(0)*n);
         at[io] = a[ii];
         at[io+1] = a[ii+1];
      }
      """). build ()

# test gpyfft using clfft
#   t2=time.time()
#   f_dev = cl.array.to_device(q,f,allocator=mem_pool)
#   f_dev = x0_dev*f_dev
#   trans1.enqueue_arrays(data=f_dev)
#   f_dev = k0_dev*f_dev
#   trans1.enqueue_arrays(data=f_dev,forward=False)
#   f_dev = x1_dev*f_dev
#   trans2.enqueue_arrays(data=f_dev)
#   f_dev = k1_dev*f_dev
#   trans2.enqueue_arrays(data=f_dev,forward=False)
#   f_dev = x2_dev*f_dev
#   trans1.enqueue_arrays(data=f_dev)
#   f_dev = k0_dev*f_dev
#   trans1.enqueue_arrays(data=f_dev,forward=False)
#   f_dev = x1_dev*f_dev
#   f_test = f_dev.get()
#   q.finish()
#   t3=time.time()
#   print("N = ",N)
#   print(f_test.dtype)
#   print("Stupid clfft ",t3-t2," seconds")
#   print("speed up ",(t1-t0)/(t3-t2))
#   
#   print("difference ",np.max(np.abs(ftmp13-f_test)))

#Now pyvkfft opencl backend with stupid transpose

   qt_app = pyvkfft.opencl.VkFFTApp((N, N), dtype=dtc, queue=q, ndim=1)
   t4=time.time()
   f_dev = cl.array.to_device(q,f,allocator=mem_pool)
   f_dev = x0_dev*f_dev
   qt_app.fft(f_dev)
   f_dev = k0_dev*f_dev
   qt_app.ifft(f_dev)
   f_dev = x1_dev*f_dev
   prg.transpose(q, (N,N), None, ft_dev.data, f_dev.data, np.uint32(N))
   qt_app.fft(ft_dev)
   ft_dev = k1t_dev*ft_dev
   qt_app.ifft(ft_dev)
   prg.transpose(q, (N,N), None, f_dev.data, ft_dev.data, np.uint32(N))
   f_dev = x2_dev*f_dev
   qt_app.fft(f_dev)
   f_dev = k0_dev*f_dev
   qt_app.ifft(f_dev)
   f_dev = x1_dev*f_dev
   f_test = f_dev.get()
   q.finish()
   t5=time.time()

   print("N = ",N)
   print(f_test.dtype)
   print("Stupid vkfft ",t5-t4," seconds")
   print("speed up ",(t1-t0)/(t5-t4))
   
   print("difference ",np.max(np.abs(ftmp13-f_test)))
   
   

