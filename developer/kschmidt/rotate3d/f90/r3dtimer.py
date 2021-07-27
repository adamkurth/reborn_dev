if __name__ == "__main__":
   import gc
   import numpy as np
   import time
   from reborn.utils import rotate3D
   from r3dks4 import rotate3D as rotate3Dpy
   from r3dks4 import rotate3Dvkfft
   import matplotlib
   try:
      matplotlib.use('pdf')
   except ValueError:
      matplotlib.use('agg')
   import matplotlib.pyplot as plt
   import scipy
   Ns = (27, 32, 50, 64, 81, 128, 150, 256, 384, 507, 512)
#   Ns = (32, 50)
   Nr = 20
   routines = [rotate3Dpy , rotate3Dvkfft]
   types = [np.complex128, np.complex64, np.float64, np.float32]
   rot_times = np.zeros((len(types)*len(routines)+1,len(Ns)),dtype=np.float64)
   Rs = scipy.spatial.transform.Rotation.random(Nr)
   iin = -1
   for N in Ns:
      iin += 1
      tempr = np.random.normal(size=(N,N,N))
      tempi = np.random.normal(size=(N,N,N))
      f3d = tempr+1j*tempi
      f3dr = tempr.copy()
      f30 = np.zeros((N,N,N),np.complex128)
      f30r = np.zeros((N,N,N),np.float64)
      ic = -1
      for t in types:
         fin = f30r
         if t in [np.complex64, np.complex128]:
            fin = f30
         for r in routines:
            r3df = r(fin.astype(t))
            ic += 1
            t0 = time.time()
            for i in range(Nr):
               r3df.f = fin.astype(t)
               r3df.rotation(Rs[i])
               f30 = r3df.f
            t1 = time.time()
            rot_times[ic,iin] = (t1-t0)/Nr
            del r3df
            gc.collect()

      ic += 1
      t0 = time.time()
      t0 = time.time()
      for i in range(Nr):
         f30 = rotate3D(f3d,Rs[i])
      t1 = time.time()
      rot_times[ic,iin] = (t1-t0)/Nr
      
   l = ["-bo", "-ro", "-go", "-yo", "-bx", "-rx", "-gx", "-yx", "-co"]
   ic = -1
   for t in types:
      for r in routines:
         ic += 1
         plt.plot(Ns,rot_times[ic,:],l[ic],label=r.__name__ + t.__name__)
   ic += 1
   plt.plot(Ns,rot_times[ic,:],l[ic],label="Joe pythone" + np.complex128.__name__)
   plt.yscale("log")
   plt.xlabel("N")
   plt.ylabel("Time(s)")
   plt.grid(True,which="both")
   plt.legend(loc='best')
   plt.savefig('time.pdf')
   plt.clf()

   for i in range(len(rot_times[:,0])):
      rot_times[i,:] = rot_times[8,:]/rot_times[i,:]

   ic = -1
   for t in types:
      for r in routines:
         ic += 1
         plt.plot(Ns,rot_times[ic,:],l[ic],label=r.__name__ + t.__name__)
   ic += 1
   plt.plot(Ns,rot_times[ic,:],l[ic],label="Joe python" + np.complex128.__name__)
   plt.xlabel("N")
   plt.yscale("linear")
   plt.ylabel("Speedup")
   plt.grid(True,which="both")
   plt.legend(loc='best')
   plt.savefig('speedup.pdf')
