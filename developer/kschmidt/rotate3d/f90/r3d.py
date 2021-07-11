
if __name__ == "__main__":
   import numpy as np
   import time
   import matplotlib.pyplot as plt
   from reborn.utils import rotate3D
   from r3df03py import rotate3dfftw
   from r3dks import rotate3D as rotate3Dks
   from r3dks import rotate3Djoeorder
   from r3dks import rotate3Dvkfft
   import scipy
   print("read in image and pad")
   t0 = time.time()
   N = 512
   fin = plt.imread('../bw.png').astype(np.float64)
   assert(fin.shape[0] == fin.shape[1])
   N0 = fin.shape[0]
   Nadd = int((N-N0)/2)
#   Nadd = int(np.ceil(np.sqrt(2)*N0/2)-N0/2)
#   Nadd += 10
#   N = N0+2*Nadd
   print("N set to ",N)
   fr = np.zeros((N,N),dtype=np.float64)
   fr[Nadd:Nadd+N0,Nadd:Nadd+N0] = fin
   f3d = np.zeros((N,N,N),dtype=np.complex128)
   f3 = np.zeros((N,N,N),dtype=np.complex128)
   for i in range(Nadd,N-Nadd):
      f3d[i,:,:] = fr
   N2 = (N/2)**2  
   i0 = np.reshape(np.outer(np.ones((N,N)),np.arange(N)),(N,N,N))
   i1 = np.transpose(i0,axes=(1,2,0))
   i2 = np.transpose(i0,axes=(2,0,1))
   truncate = (i0-N/2)**2+(i1-N/2)**2+(i2-N/2)**2 > (N/2)**2
   f3d[truncate] = 0.0
   f3a = f3d.copy()
   f3 = f3d.copy()
   euler = np.zeros(3,dtype=np.float64)
   eulerj = np.zeros(3,dtype=np.float64)
   euler[0] = 0.1+np.pi
   euler[1] = 0.22
   euler[2] = 0.35
   R = scipy.spatial.transform.Rotation.from_euler('xyx',euler)
   t1 = time.time()
   print("set up done, time = ",t1-t0," seconds")

   t1 = time.time()
   rotate3dfftw(f3.T,euler)
   t2 = time.time()
   print("fftw ",t2-t1,"seconds")

   t1 = time.time()
   f1 = rotate3D(f3d,R)
   t2 = time.time()
   print("joe ",t2-t1,"seconds")

   t1 = time.time()
   r3df = rotate3Dks(f3d)
   r3df.rotation(R)
   f2 = r3df.f
   t2 = time.time()
   print("kevin pure python plus copies",t2-t1,"seconds")

   t1 = time.time()
   r3df = rotate3Djoeorder(f3d)
   r3df.rotation(R)
   f4 = r3df.f
   t2 = time.time()
   print("joe order pure python plus copies",t2-t1,"seconds")

   t1 = time.time()
   rotate3dfftw(f3a.T,euler)
   t2 = time.time()
   print("fftw second call",t2-t1,"seconds")

   t1 = time.time()
   r3df = rotate3Dvkfft(f3d)
   t2 = time.time()
   print("vkfft create instance",t2-t1,"seconds")
   t1 = time.time()
   r3df.rotation(R)
   f5 = r3df.f
   t2 = time.time()
   print("vkfft plus copies",t2-t1,"seconds")
   del r3df # clean up opencl
   ns=3*N//4
   print("original",f3d[ns,ns,ns])
   print("joe, kevin, fftw, joeorder, vkfft ")
   print(f1[ns,ns,ns],f2[ns,ns,ns],f3[ns,ns,ns],f4[ns,ns,ns],f5[ns,ns,ns])
   print("joe-kevin",np.max(np.abs(f1-f2)))
   print("joe-fftw",np.max(np.abs(f1-f3)))
   print("kevin-fftw",np.max(np.abs(f2-f3)))
   print("kevin-vkfft",np.max(np.abs(f2-f5)))
   print("joe-joeorder",np.max(np.abs(f1-f4)))
   n2 = int(N/2)
   print("joe rotation")
   plt.imshow(np.real(f1[n2,:,:]),cmap=plt.cm.gray)
   plt.show()
   print("kevin rotation")
   plt.imshow(np.real(f2[n2,:,:]),cmap=plt.cm.gray)
   plt.show()
   print("fftw rotation")
   plt.imshow(np.real(f3[n2,:,:]),cmap=plt.cm.gray)
   plt.show()
   print("vkfft rotation")
   plt.imshow(np.real(f5[n2,:,:]),cmap=plt.cm.gray)
   plt.show()
   print("joeorder rotation")
   plt.imshow(np.real(f4[n2,:,:]),cmap=plt.cm.gray)
   plt.show()


