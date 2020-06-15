"""
Created on Tue Jun 5 14:09:16 2020

@author: Rainier
"""

import numpy as np
import scipy as sci
from reborn.utils import trilinear_insert
import matplotlib.pyplot as plt
import sys
import time
import reborn as ba
import reborn.target.crystal as crystal
import reborn.simulate.clcore as core
from reborn.simulate.examples import lysozyme_pdb_file
from reborn.utils import rotation_about_axis
from scipy import constants as const




#attempts to get our pdb file from simulate_pbd example

hc = const.h*const.c

show = True     # Display the simulated patterns
double = False  # Use double precision if available
dorotate = False  # Check if rotation matrices work
if 'view' in sys.argv:
    show = True
if 'double' in sys.argv:
    double = True
if 'dorotate' in sys.argv:
    dorotate = True
if 'noplots' in sys.argv:
    show = False

clcore = core.ClCore(group_size=32, double_precision=double)

# Create a detector
print('Setting up the detector')
tstart = time.time()
nPixels = 1001
pixelSize = 100e-6
detectorDistance = 0.5
wavelength = 1.5e-10
beam_vec = np.array([0, 0, 1])
pad = ba.detector.PADGeometry(n_pixels=nPixels, pixel_size=pixelSize, distance=detectorDistance)
q_vecs = pad.q_vecs(beam_vec=beam_vec, wavelength=wavelength)
n_pixels = q_vecs.shape[0]
tdif = time.time() - tstart
print('CPU q array created in %7.03f ms' % (tdif * 1e3))

# Load a crystal structure from pdb file
pdbFile = lysozyme_pdb_file  # Lysozyme
print('Loading pdb file (%s)' % pdbFile)
cryst = crystal.CrystalStructure(pdbFile)
r = cryst.molecule.coordinates  # These are atomic coordinates (Nx3 array)
n_atoms = r.shape[0]

# Look up atomic scattering factors (they are complex numbers)
print('Getting scattering factors')
f = ba.simulate.atoms.get_scattering_factors(cryst.molecule.atomic_numbers, hc / wavelength)


print('Generate rotation matrix')
if dorotate:
    phi = 0.5
    R = np.array([[np.cos(phi), np.sin(phi), 0], [-np.sin(phi), np.cos(phi), 0], [0, 0, 1]])
else:
    R = np.eye(3)
    
q = q_vecs
n_trials=3

if 1:

    print("Use a 3D lookup table on the GPU")
    

    res = 10e-10  # Resolution
    qmax = 2 * np.pi / res
    qmin = -qmax
    N = 200  # Number of samples
    n_atoms = r.shape[0]
    n_pixels = N ** 3
    
   

    a_map_dev = clcore.to_device(shape=(n_pixels,), dtype=clcore.complex_t)

    for i in range(0, 1):
        t = time.time()
        clcore.phase_factor_mesh(r, f, N=N, q_min=qmin, q_max=qmax, a=a_map_dev)
        tf = time.time() - t
        print('phase_factor_mesh: %7.03f ms (%d atoms; %d pixels)' % (tf * 1e3, n_atoms, n_pixels))

    print("Interpolate patterns from GPU lookup table")
    print("Amplitudes passed as GPU array")

    q_dev = clcore.to_device(q, dtype=clcore.real_t)
    a_out_dev = clcore.to_device(dtype=clcore.complex_t, shape=(pad.n_fs * pad.n_ss))
    n_atoms = 0
    n_pixels = q.shape[0]
    for i in range(0, n_trials):
        t = time.time()
        clcore.mesh_interpolation(a_map_dev, q_dev, N=N, q_min=qmin, q_max=qmax, R=R, a=a_out_dev)
        tf = time.time() - t
        print('mesh_interpolation: %7.03f ms' % (tf * 1e3))

    t = time.time()
    a = a_out_dev.get()
    tt = time.time() - t
    print("Moving amplitudes back to CPU memory in %7.03f ms" % (tt*1e3))

    if show:
        imdisp = a.reshape(pad.n_ss, pad.n_fs)
        imdisp = np.abs(imdisp) ** 2
        imdisp = np.log(imdisp + 0.1)
        plt.imshow(imdisp, interpolation='nearest', cmap='gray', origin='lower')
        plt.title('y: up, x: right, z: beam (towards you)')
        plt.show()
    print("")
















##################################################
pi=np.pi

#rotation a 3d point, p, ccw around n={a,b,c} by theta
#this does the full step by step quarterion thing.
#unnecessary due to rot()
def point_rotate(p,n,theta):
    n=n/np.sum(n)
    sx = np.array([[0, 1],[ 1, 0]])
    sy = np.array([[0, -1j],[1j, 0]])
    sz = np.array([[1, 0],[0, -1]])
    s0= np.array([[1, 0],[0, 1]])
    s=np.array([sx,sy,sz])
    def dot(n,s):
        sum=0
        for q in range(3):
            sum=n[q]*s[q]+sum
        return sum
    p=-1j*dot(p,s)
    A1=s0*np.cos(theta/2)+1j*dot(n,s)*np.sin(theta/2)
    A2=s0*np.cos(theta/2)-1j*dot(n,s)*np.sin(theta/2)
    M=np.matmul(A1,np.matmul(p,A2))
    M=np.ndarray.flatten(M)
    Dcmp=np.array([[0,1j,1j,0],[0,-1,1,0],[1j,0,0,-1j],[1,0,0,1]])*0.5
    M=np.matmul(Dcmp,M)
    return np.real(M[0:3])
  
#unnecessary due to rand_rot() 
def point_rand_rot(p):
    n=np.array([np.random.normal(),np.random.normal(),np.random.normal()])
    theta=2*np.pi*np.random.uniform()
    return point_rotate(p,n,theta)

#rotation matrix rotating a point by phi ccw wrt vector n
#as one would write it in lin alg to operate on col vecs
def R(n,phi):
    n=np.array(n)
    n=n/np.sqrt(np.sum(n**2))
    [nx,ny,nz]=n
    R=np.array([
          [0.5*(1+nx**2-ny**2-nz**2+(1-nx**2+ny**2+nz**2)*np.cos(phi)),
           nx*ny-nx*ny*np.cos(phi)-nz*np.sin(phi),
           nx*nz-nx*nz*np.cos(phi)+ny*np.sin(phi)],
            
           [nx*ny-nx*ny*np.cos(phi)+nz*np.sin(phi),
            0.5*(1-nx**2+ny**2-nz**2+(1+nx**2-ny**2+nz**2)*np.cos(phi)),
            ny*nz-ny*nz*np.cos(phi)-nx*np.sin(phi)],
             
            [nx*nz-nx*nz*np.cos(phi)-ny*np.sin(phi),
             ny*nz-ny*nz*np.cos(phi)+nx*np.sin(phi),
             0.5*(1-nx**2-ny**2+nz**2+(1+nx**2+ny**2-nz**2)*np.cos(phi))]])

    return R

#multiplying the rotaion matrix in a way that naturally perserves c order
#points is an list of points (nparray)
def rot(points,n,phi):
    return np.dot(points,R(n,phi).T)

#generates a random rotation matrix
def randR():
    n=np.array([np.random.normal(),np.random.normal(),np.random.normal()])
    n=n/np.sqrt(np.sum(n**2))
    (nx,ny,nz)=(n[0],n[1],n[2])
    phi=2*np.pi*np.random.uniform()
    R=np.array([
          [0.5*(1+nx**2-ny**2-nz**2+(1-nx**2+ny**2+nz**2)*np.cos(phi)),
           nx*ny-nx*ny*np.cos(phi)-nz*np.sin(phi),
           nx*nz-nx*nz*np.cos(phi)+ny*np.sin(phi)],
            
           [nx*ny-nx*ny*np.cos(phi)+nz*np.sin(phi),
            0.5*(1-nx**2+ny**2-nz**2+(1+nx**2-ny**2+nz**2)*np.cos(phi)),
            ny*nz-ny*nz*np.cos(phi)-nx*np.sin(phi)],
             
            [nx*nz-nx*nz*np.cos(phi)-ny*np.sin(phi),
             ny*nz-ny*nz*np.cos(phi)+nx*np.sin(phi),
             0.5*(1-nx**2-ny**2+nz**2+(1+nx**2+ny**2-nz**2)*np.cos(phi))]])

    return R

def rand_rot(points):
    out=np.dot(points,randR().T)
    return out

#returns a list of values to go with our vectors from arr
#finds the value of the voxel it falls within
# def trilinear_standin(arr,r_min,r_max,n_bin,vecs):
#     out=np.array([])
#     for n in range(len(vecs)):
#         (x,y,z)=(vecs[n][0],vecs[n][1],vecs[n][2])
#         (xmin,ymin,zmin)=(r_min[0],r_min[1],r_min[2])
#         (xmax,ymax,zmax)=(r_max[0],r_max[1],r_max[2])
#         i=int(np.floor((x-xmin)/dx+1/2))
#         j=int(np.floor((y-ymin)/dy+1/2))
#         k=int(np.floor((z-zmin)/dz+1/2))
#         if x>xmax or y>ymax or z>zmax:
#             print('error (',x,y,z,') > (',xmax,ymax,zmax,')')
#         if x<xmin or y<ymin or z<zmin:
#             print('error (',x,y,z,') > (',xmin,ymin,zmin,')')
#         out=np.append(out,arr[i,j,k])
#     return out

from reborn.target.density import trilinear_interpolation

def trilinear_standin(arr,r_min,r_max,n_bin,vecs):
    return trilinear_interpolation(arr, vecs, x_min=r_min, x_max=r_max)

#doing a single trilinear insert, with real part output
def insert(data_coord, data_val, r_min, r_max, n_bin):
        mask=np.ones_like(data_val)
        data_out,weight_out=trilinear_insert(data_coord, data_val, r_min, r_max, n_bin, mask)
        weight_out[weight_out==0]=1
        return np.real(data_out/weight_out)
    
#plots an array summed over the x axis
def plt_collapse(ar):
    plt.imshow(np.sum(ar,axis=0))

#returns N lists of poisson values,
#one value in each list for each detector point given in 'bowl'
def create_fake_data(arr,r_min,r_max,n_bin,bowl,N):
    print('creating fake data')
    fake_data=[]
    for n in range(N):
        fake_data.append(np.random.poisson(
            trilinear_standin(arr,r_min,r_max,n_bin,rand_rot(bowl))))
    print('fake data complete')
    return np.array(fake_data)

#tabulation of ln(N!)
lnfact=[]
for n in range(100000):
    lnfact.append(sci.special.gammaln(n+1))
lnfact=np.array(lnfact)
#having it work on arrays
def lnfactorial(arr):
    return lnfact[arr.astype(np.int)]

#calculating the probability of a particular exposure, 'datum',
#on a rotated detector array, 'bowl_rot',
#to be sampled from a model, 'denisty' M
def lnprob_poisson(M,r_min,r_max,n_bin,bowl_rot,datum):
    lamb=trilinear_standin(M,r_min,r_max,n_bin,bowl_rot)
    k=datum
    lnp=k*np.log(lamb)-lamb-lnfactorial(k)
    #print(np.min(lnp[lamb!=0]))
    lnp[lamb==0]=-100
    #print('lnp',np.sum(lnp))
    return np.sum(lnp)
    
#prop dist as a gaussian cut off at 0 and 2 pi
#irrelavant currently
def prob_guass_periodic(phi,sigma):
    assert phi<2*pi and phi>=0
    nrmlz=np.sqrt(2*pi)*sigma*sci.special.erf(pi/np.sqrt(2)/sigma)
    return np.exp(-0.5*(phi/sigma)**2)/nrmlz

#sampling the periodic gaussian in a dumb way
#samples angles from -pi to pi
def sample_gauss_periodic(sigma):
    phi_p=sigma*np.random.randn()
    while phi_p>pi or phi_p<-pi:
        #print('overshot',phi_p)
        phi_p=sigma*np.random.randn()
    return phi_p

#'data' is a list of the exposures
#'dect_arr' are the detector pixel q-vector coordinates
#'model_input' can be any fourier density but eventually should be noise
def emmc(data,dect_arr,model_input,r_min,r_max,n_bin,N_remodels,N_metropolis,sigma):
    print('beginning EMMC')
    t=time.time()
    model=model_input
    models=[model]
    def lnp(model,points,datum):
        return lnprob_poisson(model,r_min,r_max,n_bin,points,datum)
    for v in range(N_remodels):
        model_val=np.zeros_like(model_input)
        model_weights=np.zeros_like(model_input)
        for d in range(np.shape(data)[0]):
            datum=data[d]
            da=np.array(rand_rot(dect_arr))
            accpt=0
            for l in range(N_metropolis):
                n_vec=np.array([np.random.normal(),np.random.normal(),np.random.normal()])
                n_vec/=np.sqrt(np.sum(n_vec**2))
                phi=sample_gauss_periodic(sigma)
                da_prop=rot(da,n_vec,phi)
                lnA=np.min([0,lnp(model,da_prop,datum)-lnp(model,da,datum)])
                u=np.random.uniform()
                if np.log(u)<=lnA:
                    da=da_prop
                    accpt+=1
                mask=np.ones_like(datum)
                (mv,mw)=trilinear_insert(da, datum, r_min, r_max, n_bin, mask)
                model_val+=np.real(mv)
                model_weights+=np.real(mw)
                #print('model',v+1,'datum',d,'metro',l+1)
            print('    datum ',d,"complete",'accpt',100*accpt/N_metropolis)
        model_weights[model_weights==0]=1
        model=model_val/model_weights
        print("remodel ",v,"complete")
        models.append(model)
    tf=time.time()-t
    print("emmc completed in",tf, "sec")
    return models
#######################################################              
                
            













#lets generate some points on a z oriented ewald bowl to rotate around
#our bowl's radius, K
K=2.5
xmax_grid=4
xmin_grid=-xmax_grid
#dx_grid=1
#N_grid=(xmax_grid-xmin_grid)/dx+1
N_grid=50
dx_grid=(xmax_grid-xmin_grid)/(N_grid-1)
bowl=[]
for i in range(int(N_grid)):
    for j in range(int(N_grid)):
        x_grid=i*dx_grid+xmin_grid
        y_grid=j*dx_grid+xmin_grid
        bowl.append([x_grid,y_grid])
bowl=np.array(bowl)
bowl2=np.array([[]])
for i in range(len(bowl)):
    p=bowl[i]
    if ((K**2-p[0]**2-p[1]**2)>=0)==True:
        v=np.append(bowl[i],np.sqrt(K**2-p[0]**2-p[1]**2)-K)
        bowl2=np.append(bowl2,v)
bowl=np.reshape(bowl2,[-1,3])
#which give us a list points points on an ewald sphere named 'bowl'

#now lets generate a fake fourier pattern, fourier_test
#here's a gaussian funct
def funct(x,y,z):
    return np.exp(-(x**2+y**2+z**2))
#now to evaluate it on an arbitrarily shaped array
#to convert for int indices to spatial coordiantes we need
xmax,ymax,zmax=4,4,4
(xmin,ymin,zmin)=(-xmax,-ymax,-zmax)
Nx,Ny,Nz=50,50,50
dx,dy,dz=(xmax-xmin)/(Nx-1),(ymax-ymin)/(Ny-1),(zmax-zmin)/(Nz-1)
r_max=np.array([xmax,ymax,zmax])
r_min=np.array([xmin,ymin,zmin])
n_bin=np.array([Nx,Ny,Nz])
fourier_test=np.empty([Nx,Ny,Nz])
for i in range(len(fourier_test)):
    for j in range(len(fourier_test[0])):
        for k in range(len(fourier_test[0][0])):
            (x,y,z)=(xmin+i*dx,ymin+j*dy,zmin+k*dz)
            fourier_test[i][j][k]=funct(x,y,z-1)    
#which gives a test density map named fourier_test
flat_test=np.ones(n_bin)
#give a flat density
fourier_test2=np.empty([Nx,Ny,Nz])
for i in range(len(fourier_test)):
    for j in range(len(fourier_test[0])):
        for k in range(len(fourier_test[0][0])):
            (x,y,z)=(xmin+i*dx,ymin+j*dy,zmin+k*dz)
            fourier_test2[i][j][k]=2*funct(x*2,y*2-1.75,z*2-1.5)+2*funct(x*2,y*2+1.75,z*2-1.5)+3*funct(x*2-1.5,y*2,z*2+1.5)+2*funct(x,y,z-1.5)
#gives a density with more features

fake_data=create_fake_data(200*fourier_test2,r_min,r_max,n_bin, bowl, 10)

#using trlinear insert
#data_coord=rot(bowl,[0,1,0],pi/2)
data_coord=rot(rot(bowl,[0,1,0],pi/2),[1,0,0],pi/4)
#data_val=trilinear_standin(fourier_test,r_min,r_max,n_bin,data_coord)
#data_val=fake_data[0]
data_val=trilinear_standin(flat_test,r_min,r_max,n_bin,data_coord)
ar=insert(data_coord, data_val, r_min, r_max, n_bin)


#inserting 2 ewald bowls
'''
data_tot=np.zeros(n_bin,dtype="complex128")
weight_tot=np.zeros(n_bin,dtype="complex128")
for l in range(2):
    data_coord2=rand_rot(bowl)
    data_val2=trilinear_standin(fourier_test,r_min,r_max,n_bin,data_coord2)
    #data_val2=trilinear_standin(flat_test,r_min,r_max,n_bin,data_coord2)
    mask2=np.ones_like(data_val2)
    data_out2,weight_out2=trilinear_insert(data_coord2, data_val2, r_min, r_max, n_bin, mask)
    data_tot+=data_out2
    weight_tot+=weight_out2
weight_tot[weight_tot==0]=1
data_tot/=weight_tot
bowl_arr2=data_tot

#plot cross sections to show it works
plt.imshow(np.real(bowl_arr2[int(np.floor(Nx*1/2))]))
'''


#Checking to see if our proposal dist works
'''
smpls=[]
for v in range(100000):
    smpls.append(sample_gauss_periodic(pi/2.5))
    
plt.hist(smpls,bins=2*pi*np.array(range(0,100))/100-pi)
'''
#seems correct

models=emmc(fake_data,bowl,fourier_test2,r_min,r_max,n_bin,1,10,pi/2)
plt.imshow(models[-1][int(np.floor(Nx*1/2))])



print("done")