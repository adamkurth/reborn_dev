import time
import numpy as np
import scipy
import scipy.spatial
from r3dks4 import rotate3D as rotate3Dpy
from r3dks4 import rotate3Dvkfft as rotate3Dv
from r3dks4 import rotate3Dlegacy as rotate3Dl
from reborn.utils import rotate3D as rotate3Dj
import matplotlib.pyplot as plt
import skimage
import skimage.measure
import mpl_toolkits


def makegaussians(w,g,s,N):
   x = np.linspace(-0.5*(1.0-1.0/N),0.5*(1.0-1.0/N),num=N)
   d = np.zeros((N,N,N),dtype=np.float64)
   for ig in range(len(w)):
      xgauss = np.exp(-0.5*((x-g[ig,0])/s)**2)
      ygauss = np.exp(-0.5*((x-g[ig,1])/s)**2)
      zgauss = np.exp(-0.5*((x-g[ig,2])/s)**2)
      d += w[ig]*np.tile(np.reshape(xgauss,(N,1,1)),(1,N,N))*\
         np.tile(np.reshape(ygauss,(1,N,1)),(N,1,N))*\
         np.tile(np.reshape(zgauss,(1,1,N)),(N,N,1))
   return d


rng = np.random.default_rng(1717171717)
Nr = 10
Rs = scipy.spatial.transform.Rotation.random(Nr,random_state=rng)
Ngr = 8 
Ngi = 8
sigma = 0.05
gr0 = (rng.random((Ngr,3))-0.5)*sigma*3.0
wr = rng.random(Ngr)-0.5
gi0 = (rng.random((Ngi,3))-0.5)*sigma*3.0
wi = rng.random(Ngi)-0.5
Ns = [7, 16, 27, 32, 48, 64, 75]
N = 32
sp = (1.0/N,1.0/N,1.0/N)
#gr0 = np.zeros((6,3),dtype=np.float64)
#wr = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
#gr0[0,:] = (0.0,0.0,0.2)
#gr0[1,:] = (0.0,0.3,0.0)
#gr0[2,:] = (-0.4,0.0,0.0)
#gr0[3,:] = (0.4,0.0,0.0)
#gr0[4,:] = (0.0,-0.3,0.0)
#gr0[5,:] = (0.0,0.0,-0.2)
datar = makegaussians(wr,gr0,sigma,N)
mx = np.max(datar)
mn = np.min(datar)
print(mx,mn)
lev = 0.02
verts, faces, normals, values = skimage.measure.marching_cubes(datar,\
   spacing=sp,level=lev)
fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(1,2,1, projection='3d')
mesh = mpl_toolkits.mplot3d.art3d.Poly3DCollection(verts[faces])
mesh.set_edgecolor('k')
ax.add_collection3d(mesh)

r3df = rotate3Dv(datar)
r3df.rotation(Rs[0])
rotgrid = r3df.f
gr = Rs[0].apply(gr0)
rotgaus = makegaussians(wr,gr,sigma,N)

lev = 0.02
ax = fig.add_subplot(1,2,2, projection='3d')
verts, faces, normals, values = skimage.measure.marching_cubes(rotgrid,\
   spacing=sp,level=lev)
mesh = mpl_toolkits.mplot3d.art3d.Poly3DCollection(verts[faces])
mesh.set_edgecolor('k')
ax.add_collection3d(mesh)
verts, faces, normals, values = skimage.measure.marching_cubes(rotgaus,\
   spacing=sp,level=lev)
mesh = mpl_toolkits.mplot3d.art3d.Poly3DCollection(verts[faces])
mesh.set_edgecolor('r')
ax.add_collection3d(mesh)
plt.tight_layout()
plt.show()
