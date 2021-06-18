
if __name__ == "__main__":
   import numpy as np
   import time
   import matplotlib.pyplot as plt
   from reborn.utils import rotate3D
   from r3df03py import rotate3dfftw
   from r3dks import rotate3Dks
   
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
   for i in range(N):
      f3d[i,:,:] = fr
      f3[i,:,:] = fr
   
#   euler = (0.1, 0.0, 0.0)
#   eulerj = (-0.1, 0.0, 0.0)
#   euler = (5.1, 0.0, 0.0)
#   eulerj = (-5.1, 0.0, 0.0)
   euler = np.zeros(3,dtype=np.float64)
   eulerj = np.zeros(3,dtype=np.float64)
   euler[0] = 0.1+np.pi
   euler[1] = 0.17
   euler[2] = 0.35
   eulerj[:] = euler[:]
   eulerj[0] = -eulerj[0]
   eulerj[2] = -eulerj[2]

# For these to be identical, with no negative signs, change joe's code to
# f_rot[ii, :, :] = np.transpose(rotate2D(np.transpose(f[ii, :, :]), kxfac, xfac, kyfac, yfac, n90_mod_Four))
   t1 = time.time()
   rotate3dfftw(f3.T,euler)
   t2 = time.time()
   f1 = rotate3D(f3d,eulerj)
   t3 = time.time()
   f2 = rotate3Dks(f3d,euler)
   t4 = time.time()
   for i in range(N):
      f3[i,:,:] = fr
   t5 = time.time()
   rotate3dfftw(f3.T,euler)
   t6 = time.time()
   ns=3*N//4
   print("original",f3d[ns,ns,ns])
   print(f1[ns,ns,ns],f2[ns,ns,ns],f3[ns,ns,ns])
   print("joe ",t3-t2,"seconds")
   print("kevin ",t4-t3,"seconds")
   print("fftw ",t2-t1,"seconds")
   print("fftw second time ",t6-t5,"seconds")
   
   print("joe rotation")
   plt.imshow(np.real(f1[ns,:,:]),cmap=plt.cm.gray)
   plt.show()
   print("kevin rotation")
   plt.imshow(np.real(f2[ns,:,:]),cmap=plt.cm.gray)
   plt.show()
   print("fftw rotation")
   plt.imshow(np.real(f3[ns,:,:]),cmap=plt.cm.gray)
   plt.show()
