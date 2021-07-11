
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
   
   fin = plt.imread('../bw.png').astype(np.float64)
   assert(fin.shape[0] == fin.shape[1])
   N0 = fin.shape[0]
   Nadd = int(np.ceil(np.sqrt(2)*N0/2)-N0/2)
   Nadd += 10
   N = N0+2*Nadd
   print("N",N)
   fr = np.zeros((N,N),dtype=np.float64)
   fr[Nadd:Nadd+N0,Nadd:Nadd+N0] = fin
   f3d = np.zeros((N,N,N),dtype=np.complex128)
   f3 = np.zeros((N,N,N),dtype=np.complex128)
   for i in range(Nadd,N-Nadd):
      f3d[i,:,:] = fr
      f3[i,:,:] = fr
   
   euler = np.zeros(3,dtype=np.float64)
   eulerj = np.zeros(3,dtype=np.float64)
   euler[0] = 0.1+np.pi
   euler[1] = 0.70
   euler[2] = 0.35
   eulerj[:] = -euler[:]
   R = scipy.spatial.transform.Rotation.from_euler('xyx',eulerj)

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

   for i in range(Nadd,N-Nadd):
      f3[i,:,:] = fr
   t1 = time.time()
   rotate3dfftw(f3.T,euler)
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
   ns=3*N//4
   print("original",f3d[ns,ns,ns])
   print("joe, kevin, fftw, joeorder, vkfft ")
   print(f1[ns,ns,ns],f2[ns,ns,ns],f3[ns,ns,ns],f4[ns,ns,ns],f5[ns,ns,ns])
   print("joe-kevin",np.max(np.abs(f1-f2)))
   print("joe-fftw",np.max(np.abs(f1-f3)))
   print("kevin-fftw",np.max(np.abs(f2-f3)))
   print("kevin-vkfft",np.max(np.abs(f2-f5)))
   print("joe-joeorder",np.max(np.abs(f1-f4)))

   print("joe rotation")
   plt.imshow(np.real(f1[ns,:,:]),cmap=plt.cm.gray)
   plt.show()
   print("kevin rotation")
   plt.imshow(np.real(f2[ns,:,:]),cmap=plt.cm.gray)
   plt.show()
   print("fftw rotation")
   plt.imshow(np.real(f3[ns,:,:]),cmap=plt.cm.gray)
   plt.show()
   print("vkfft rotation")
   plt.imshow(np.real(f5[ns,:,:]),cmap=plt.cm.gray)
   plt.show()
   print("joeorder rotation")
   plt.imshow(np.real(f4[ns,:,:]),cmap=plt.cm.gray)
   plt.show()


