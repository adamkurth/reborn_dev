"""
Created on Tue Jun 5 14:09:16 2020

@author: Rainier
"""

import time
import datetime as dt
import os
import numpy as np
import scipy as sci
from reborn.utils import trilinear_insert
from reborn.utils import rotate3D
from reborn.target.density import trilinear_interpolation, trilinear_insertion
from scipy import constants as const
import matplotlib.pyplot as plt
np.random.seed(2020)
import reborn
import reborn.target.crystal as crystal
import reborn.simulate.clcore as core


##############################################
#Simulating Lysase
#see https://kirianlab.gitlab.io/reborn/auto_examples/plot_simulate_pdb.html#sphx-glr-auto-examples-plot-simulate-pdb-py

eV = const.value('electron volt')
r_e = const.value('classical electron radius')

#setting up the gpu
simcore = core.ClCore(group_size=32, double_precision=False)

# Let's check which device we are using:
print(simcore.get_device_name())

# First we set up a pixel array detector, PAD and an x-ray, Beam:
#beam = reborn.source.Beam(photon_energy=10000*eV, diameter_fwhm=0.2e-6, pulse_energy=2)
beam = reborn.source.Beam(photon_energy=5000*eV, diameter_fwhm=0.2e-6, pulse_energy=0.02)
fluence = beam.photon_number_fluence
pad = reborn.detector.PADGeometry(shape=(51, 51), pixel_size=4000e-6, distance=0.5)
#pad = reborn.detector.PADGeometry(shape=(1001, 1001), pixel_size=100e-6, distance=0.5)
q_vecs = pad.q_vecs(beam=beam)
n_pixels = q_vecs.shape[0]
solid_angles = pad.solid_angles()
polarization_factors = pad.polarization_factors(beam=beam)
q_mags = pad.q_mags(beam=beam)

# Next we load a crystal structure from pdb file, from which we can get coordinates and scattering factors.
cryst = crystal.CrystalStructure('2LYZ')
r_vecs = cryst.molecule.coordinates  # These are atomic coordinates (Nx3 array)
atomic_numbers = cryst.molecule.atomic_numbers
n_atoms = r_vecs.shape[0]

'''
#doing the full sum in a slow way, idk
uniq_z = np.unique(atomic_numbers)
grouped_r_vecs = []
grouped_fs = []
for z in uniq_z:
    subr = np.squeeze(r_vecs[np.where(atomic_numbers == z), :])
    grouped_r_vecs.append(subr)
    grouped_fs.append(reborn.simulate.atoms.hubbel_henke_scattering_factors(q_mags=q_mags, photon_energy=beam.photon_energy,
                                                            atomic_number=z))
#calulating the intesities for a set of q vectors
intensities = []
sa = solid_angles
p = polarization_factors
q = q_vecs
amps = 0
for j in range(len(grouped_fs)):
    f = grouped_fs[j]
    r = grouped_r_vecs[j]
    a = simcore.phase_factor_qrf(q, r)
    amps += a*f
ints = r_e**2*fluence*sa*p*np.abs(amps)**2
intensities.append(pad.reshape(ints))

# Let's see how many photons hit the detector:
print('# photons total: %d' % np.round(np.sum(reborn.detector.concat_pad_data(intensities))))
'''

# Look up atomic scattering factors (they are complex numbers).  Note that these are not :math:`q`-dependent; they are
# the forward scattering factors :math:`f(0)`.
f = cryst.molecule.get_scattering_factors(beam=beam)

# Pre-allocation of GPU arrays
q_dev = simcore.to_device(q_vecs, dtype=simcore.real_t)
r_dev = simcore.to_device(r_vecs, dtype=simcore.real_t)
f_dev = simcore.to_device(f, dtype=simcore.complex_t)
a_dev = simcore.to_device(shape=(q_dev.shape[0]), dtype=simcore.complex_t)


# Speeding up with lookup tables
# First we compute the 3D mesh of diffraction amplitudes:
q_max = np.max(pad.q_mags(beam=beam))
N =40  # Number of samples(bins)
a_map_dev = simcore.to_device(shape=(N ** 3,), dtype=simcore.complex_t)
simcore.phase_factor_mesh(r_vecs, f, N=N, q_min=-q_max, q_max=q_max, a=a_map_dev)

#voila the lysozyme 3d scattering intensity
lys_intensity=np.ndarray.astype(np.abs(np.reshape(a_map_dev.get(),[N,N,N])),'float64')
F_lys2=np.ndarray.astype(np.abs(np.reshape(a_map_dev.get(),[N,N,N]))**2,'float64')






##################################################
pi=np.pi

#rounding to the nearsest second for datetime
#to get current time
#round_sec(dt.datetime.today())
def round_sec(date_time):
    rounded = date_time
    if rounded.microsecond >= 500000:
        rounded = rounded + dt.timedelta(seconds=1)
    return rounded.replace(microsecond=0)
    
def now():
    old=str(round_sec(dt.datetime.today()))
    new=''
    for s in old:
        if s==':':
            new+='.'
        else:
            new+=s
    return new

    

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
#(^commented out and slow)
#using Rick's from target density
def trilinear_standin(arr,q_min,q_max,vecs):
    '''
    out=np.array([])
    for n in range(len(vecs)):
        (x,y,z)=(vecs[n][0],vecs[n][1],vecs[n][2])
        (xmin,ymin,zmin)=(q_min[0],q_min[1],q_min[2])
        (xmax,ymax,zmax)=(q_max[0],q_max[1],q_max[2])
        i=int(np.floor((x-xmin)/dx+1/2))
        j=int(np.floor((y-ymin)/dy+1/2))
        k=int(np.floor((z-zmin)/dz+1/2))
        if x>xmax or y>ymax or z>zmax:
            print('error (',x,y,z,') > (',xmax,ymax,zmax,')')
        if x<xmin or y<ymin or z<zmin:
            print('error (',x,y,z,') > (',xmin,ymin,zmin,')')
        out=np.append(out,arr[i,j,k])
    return out
    '''
    #we have a proper fast interpolate now
    return trilinear_interpolation(arr, vecs, x_min=q_min, x_max=q_max)

#doing a single trilinear insert for tests, with real part output
def insert(data_coord, data_val, q_min, q_max, N_bin):
        mask=np.ones_like(data_val)
        data_out,weight_out=trilinear_insert(data_coord, data_val, q_min, q_max, N_bin, mask)
        weight_out[weight_out==0]=1
        return np.real(data_out/weight_out)
    

#plots an array summed over the x axis
def imshow_collapse(ar):
    plt.imshow(np.sum(ar,axis=0))

#plot cross section natural log through equator
#replaces values that are small compared to the avg with the average
#this eliminates the zero padding to help with contrast
def imshow_ln(ar,avg=True):
    x=ar[int(np.floor(len(ar)*0.5))]
    #x[x==x[0,0]]=np.average(x)
    if avg==True:
        x[x<np.average(x)/10**7]=np.average(x)
    plt.imshow(np.log(x),cmap='gray')
    
#plot cross section through equator
def imshow_x(ar):
    plt.imshow(ar[int(np.floor(len(ar)*0.5))],cmap='gray')


#returns N lists of poisson values,
#one value in each list for each detector point given in 'bowl'
#track=True will return the fake data and the rotations used
def create_fake_data(arr,q_min,q_max,bowl,N,track=False):
    t=time.time()
    print('creating fake data')
    fake_data=[]
    Rs=[]
    if track==False:
        for n in range(N):
            fake_data.append(np.random.poisson(trilinear_standin(arr,q_min,q_max,rand_rot(bowl))))
        print('fake data complete',time.time()-t, 'sec')
        return np.array(fake_data)
    else: 
        print('track')
        for n in range(N):
            R=randR()
            Rs.append(R)
            bowl_rot=np.dot(bowl,R.T)
            fake_data.append(np.random.poisson(trilinear_standin(arr,q_min,q_max,bowl_rot)))
        print('fake data complete',time.time()-t, 'sec')
        return np.array(fake_data),np.array(Rs)
    
#returns N lists of poisson values,
#one value in each list for each detector point given in 'bowl'
#track=True will return the fake data and the rotations used
def create_fake_data_sa(arr,q_min,q_max,q_vecs,solid_angles,N,track=False):
    t=time.time()
    print('creating fake data')
    fake_data=[]
    if track==False:
        for n in range(N):
            q_vecs_rot=rand_rot(q_vecs)
            val=trilinear_standin(arr,q_min,q_max,q_vecs_rot)
            val*=solid_angles
            fake_data.append(np.random.poisson(val))
        print('fake data complete',time.time()-t, 'sec')
        return np.array(fake_data)
    else: 
        Rs=[]
        print('track')
        for n in range(N):
            R=randR()
            Rs.append(R)
            q_vecs_rot=np.dot(q_vecs,R.T)
            val=trilinear_standin(arr,q_min,q_max,q_vecs_rot)
            val*=solid_angles
            fake_data.append(np.random.poisson(val))
        print('fake data complete',time.time()-t, 'sec')
        return np.array(fake_data),np.array(Rs)

#plotting probability over phi
#simulates an exposure from 'model_correct', and calculates the probability
#from 'model' around a vector 'n_vec' plotted with phi
def plot_prob_phi(model,model_correct,dect_arr,N_angle_samples,n_vec,log=True,zoom=False,save=False):
    datum=np.random.poisson(trilinear_standin(model_correct,q_min,q_max,dect_arr))
    n_vec=np.array(n_vec).astype('float64')
    n_vec=n_vec/np.sqrt(np.sum(n_vec**2))
    lnp_vs_phi=[]
    for j in range(N_angle_samples):
        bowl_rot=rot(dect_arr, n_vec, 2*pi*j/N_angle_samples-pi)
        lnp_vs_phi.append(lnprob_poisson(model,q_min,q_max,N_bin,bowl_rot,datum))
    lnp_vs_phi=np.array(lnp_vs_phi)
    x_values=np.array(range(N_angle_samples))*2/N_angle_samples-1
    if log==True:
        plt.plot(x_values,lnp_vs_phi)
    else:
        lnp_vs_phi_expnorm=np.exp(lnp_vs_phi-np.max(lnp_vs_phi))
        if zoom==False:
            plt.plot(x_values,lnp_vs_phi_expnorm)
        else:
            plt.plot(x_values[lnp_vs_phi_expnorm>0.00001],lnp_vs_phi_expnorm[lnp_vs_phi_expnorm>0.00001])
    if save==True:
        title='around '+str(n_vec)
        if zoom==True:
            title+=' zoom'
        if log==True:
            title+=' log'
        plt.title(title)
        now=str(round_sec(dt.datetime.today()))
        plt.savefig('EMMC_plots/rotation_prob '+title+now+'.pdf')

def max_magnitude(vecs):
    mag=0
    for vec in vecs:
        v=0
        for x in vec:
            v+=x**2
        if v>mag:
            mag=v
    return np.sqrt(mag)

def inertia_tensor(rho,qmin,qmax):
    N_bin=np.shape(rho)
    x_ = np.linspace(qmin[0],qmax[0],N_bin[0])
    y_ = np.linspace(qmin[1],qmax[1],N_bin[1])
    z_ = np.linspace(qmin[2],qmax[2],N_bin[2])
    x,y,z = np.meshgrid(x_,y_,z_,indexing='ij')
    Ixx = np.sum((y**2 + z**2)*rho)
    Iyy = np.sum((x**2 + z**2)*rho)
    Izz = np.sum((x**2 + y**2)*rho)
    Ixy = -np.sum(x*y*rho)
    Iyz = -np.sum(y*z*rho)
    Ixz = -np.sum(x*z*rho)
    I = np.array([[Ixx, Ixy, Ixz],
                  [Ixy, Iyy, Iyz],
                  [Ixz, Iyz, Izz]])
    return I


#aligns rho to rho_correct
#if return_error=True if returns the error 
#of rho compared to rho_correct once it tries to align them
def principal_axes_align(rho,rho_correct,q_min,q_max,return_error=False):
    t=time.time()
    
    #we need to turn [i,j,k] into q-coord vectors to rotate them
    #truncated to the ball r >=qmax to avoid going out of bounds
    N_bin=np.array(np.shape(rho))
    dx=(q_max[0]-q_min[0])/(N_bin[0]-1)
    dy=(q_max[1]-q_min[1])/(N_bin[1]-1)
    dz=(q_max[2]-q_min[2])/(N_bin[2]-1)
    coord_list=[]
    mask=[]
    iterator=np.nditer(rho,flags=['multi_index'],order='C')
    print('creating list of x,y,z coordinates by iterating over rho')
    for l in iterator:
        x=q_min[0]+dx*iterator.multi_index[0]
        y=q_min[1]+dy*iterator.multi_index[1]
        z=q_min[2]+dz*iterator.multi_index[2]
        mag=np.sqrt(x**2+y**2+z**2)
        if  mag<=np.min(np.abs(q_max)) and mag <=np.min(np.abs(q_min)):
            coord_list.append((x,y,z))
            mask.append(1)
        else:
            mask.append(0)
    mask=np.array(mask)
    
    #truncating rho_correct values to the ball
    rho_correct_flat=np.ndarray.flatten(rho_correct)
    rho_correct_flat[mask==0]=0
    rho_correct=np.reshape(rho_correct_flat,np.shape(rho_correct))
    rho_trunc=np.ndarray.flatten(rho)[mask==1]
        
    #Calculate the principal axes and order them by ascending eignevalue
    I=inertia_tensor(rho_correct, q_min, q_max)
    I_p=inertia_tensor(rho, q_min, q_max)
    eigvals,R = np.linalg.eigh(I)
    eigvals_p,R_p = np.linalg.eigh(I_p)
    
    #rotating rho by all enantomers
    signs=[-1,1]
    rho_rots=[]
    errors=[]
    for sign1 in signs:
        for sign2 in signs:
            for sign3 in signs:
                a=sign1*np.array([1,0,0])
                b=sign2*np.array([0,1,0])
                c=sign3*np.array([0,0,1])
                flip=np.array([a,b,c])
                rotation=np.dot(R_p,np.dot(flip,R.T))
                coord_list_rot=np.dot(coord_list,rotation)
                RHO=insert(coord_list_rot, rho_trunc, q_min, q_max, N_bin)
                rho_rots.append(RHO)
                errors.append(np.sum((rho_correct-RHO)**2))
    errors=np.array(errors)
    print('out of errors')
    print(errors)
    print('we choose',np.min(errors))
    error=np.min(errors)
    rho_rot=rho_rots[np.argmin(errors)]
    print(time.time()-t,'secs')
    if return_error==True:
        return error
    else:
        return rho_rot
    
    
#aligns rho to rho_correct
#returns all 8 flips and twists
def align_xyz(rho,q_min,q_max):
    
    #we need to turn [i,j,k] into q-coord vectors to rotate them
    #truncated to the ball r >=qmax to avoid going out of bounds
    N_bin=np.array(np.shape(rho))
    dx=(q_max[0]-q_min[0])/(N_bin[0]-1)
    dy=(q_max[1]-q_min[1])/(N_bin[1]-1)
    dz=(q_max[2]-q_min[2])/(N_bin[2]-1)
    coord_list=[]
    mask=[]
    iterator=np.nditer(rho,flags=['multi_index'],order='C')
    print('creating list of x,y,z coordinates by iterating over rho')
    for l in iterator:
        x=q_min[0]+dx*iterator.multi_index[0]
        y=q_min[1]+dy*iterator.multi_index[1]
        z=q_min[2]+dz*iterator.multi_index[2]
        mag=np.sqrt(x**2+y**2+z**2)
        if  mag<=np.min(np.abs(q_max)) and mag <=np.min(np.abs(q_min)):
            coord_list.append((x,y,z))
            mask.append(1)
        else:
            mask.append(0)
    mask=np.array(mask)
    rho_trunc=np.ndarray.flatten(rho)[mask==1]
        
    #Calculate the principal axes and order them by ascending eignevalue
    I=inertia_tensor(rho, q_min, q_max)
    eigvals,R = np.linalg.eigh(I)
    print(eigvals)
    print(R)
    
    #rotating rho by all enantomers
    signs=[-1,1]
    rho_rots=[]
    for sign1 in signs:
        for sign2 in signs:
            for sign3 in signs:
                a=sign1*np.array([1,0,0])
                b=sign2*np.array([0,1,0])
                c=sign3*np.array([0,0,1])
                flip=np.array([a,b,c])
                rotation=np.dot(R.T,flip)
                coord_list_rot=np.dot(coord_list,rotation)
                RHO=insert(coord_list_rot, rho_trunc, q_min, q_max, N_bin)
                rho_rots.append(RHO)
    return rho_rots

#truncates to the ball < qmax
def rotate_density_map(rho,R,q_min,q_max):
    #we need to turn [i,j,k] into q-coord vectors to rotate them
    #truncated to the ball r >=qmax to avoid going out of bounds
    N_bin=np.array(np.shape(rho))
    dx=(q_max[0]-q_min[0])/(N_bin[0]-1)
    dy=(q_max[1]-q_min[1])/(N_bin[1]-1)
    dz=(q_max[2]-q_min[2])/(N_bin[2]-1)
    coord_list=[]
    mask=[]
    iterator=np.nditer(rho,flags=['multi_index'],order='C')
    for l in iterator:
        x=q_min[0]+dx*iterator.multi_index[0]
        y=q_min[1]+dy*iterator.multi_index[1]
        z=q_min[2]+dz*iterator.multi_index[2]
        mag=np.sqrt(x**2+y**2+z**2)
        if  mag<=np.min(np.abs(q_max)) and mag <=np.min(np.abs(q_min)):
            coord_list.append((x,y,z))
            mask.append(1)
        else:
            mask.append(0)
    mask=np.array(mask)
    rho_trunc=np.ndarray.flatten(rho)[mask==1]
    avg1=np.average(rho_trunc)
    coord_list_rot=np.dot(coord_list,R.T)
    rho_rot=insert(coord_list_rot, rho_trunc, q_min, q_max, N_bin)
    avg2=np.average(np.ndarray.flatten(rho_rot)[mask==1])
    
    return rho_rot*avg1/avg2

#throws N rotations, takes the best.
def error_search_random(construct,truth,q_min,q_max,N):
    print('error search')
    truth=rotate_density_map(truth, randR(), q_min, q_max)
    best=truth
    error=np.sum((construct-truth)**2)
    for l in range(N):
        n=np.array([np.random.normal(),np.random.normal(),np.random.normal()])
        n=n/np.sqrt(np.sum(n**2))
        phi=np.random.uniform(low=-pi,high=pi)
        A=R(n,phi)
        truth_rot=rotate_density_map(truth, A, q_min, q_max)
        error_p=np.sum((construct-truth_rot)**2)
        print(error_p)
        if error_p<error:
            best=truth_rot
            error=error_p
            print('accpt')
    return error,best

#creating an icosohedron of points 20
gold=(1+np.sqrt(5))/2
icoso=[]
plusminus=[-1,1]
for a in plusminus:
    for b in plusminus:
        for c in plusminus:
            icoso.append([a*gold,b*gold,c*gold])
for i in plusminus:
    for j in plusminus:
        icoso.append([0,i*gold**2,j])
for i in plusminus:
    for j in plusminus:
        icoso.append([i*gold**2,j,0])
for i in plusminus:
    for j in plusminus:
        icoso.append([i,0,j*gold**2])
       
#the icosohedron identified with antipodal points 10
ics_1=[]
for i in plusminus:
    for j in plusminus:
            ics_1.append([i*gold,j*gold,gold])
for i in plusminus:
    ics_1.append([0,i*gold**2,1])
for i in plusminus:
    ics_1.append([i*gold**2,1,0])
for i in plusminus:
        ics_1.append([i,0,gold**2])
        
#creating a buckyball identifying antipodal 30
buck=[]
for i in plusminus:
    buck.append([0,i*1,3*gold])
for i in plusminus:
    buck.append([i*1,3*gold,0])
for i in plusminus:
    buck.append([i*3*gold,0,1])
for i in plusminus:
    for j in plusminus:
        buck.append([i*1,j*(2+gold),2*gold])
for i in plusminus:
    for j in plusminus:
        buck.append([i*(2+gold),j*2*gold,1])
for i in plusminus:
    for j in plusminus:
        buck.append([i*2*gold,j*1,(2+gold)])
for i in plusminus:
    for j in plusminus:
        buck.append([i*gold,j*2,gold**3])
for i in plusminus:
    for j in plusminus:
        buck.append([i*2,j*gold**3,gold])
for i in plusminus:
    for j in plusminus:
        buck.append([i*gold**3,j*gold,2])


#checks N rotations around the vertices of a buckyball
def error_search_buck(construct,truth,q_min,q_max,N,return_rotation=False,return_both=False):
    print('buckyball error search')
    construct=(construct+np.flip(construct))/2
    #get rid of any bias and truncate to the sphere
    Aint=randR()
    truth=rotate_density_map(truth, Aint, q_min, q_max)
    error=np.sum((construct-truth)**2)
    for n in buck:
        for l in range(N):
            phi=l*pi/N
            A=R(n,phi)
            truth_rot=rotate_density_map(truth, A, q_min, q_max)
            error_p=np.sum((construct-truth_rot)**2)
            if error_p<error:
                error=error_p
                print(error_p)
                Abest=np.dot(A,Aint)
    if return_rotation==True:
        return Abest
    if return_both==True:
        return np.sqrt(error/np.sum(truth**2)),Abest
    return np.sqrt(error/np.sum(truth**2))


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

#tabulation of ln(N!)
lnfact=[]
for n in range(1000000):
    lnfact.append(sci.special.gammaln(n+1))
lnfact=np.array(lnfact)
#having it work on arrays
def lnfactorial(arr):
    return lnfact[arr.astype(np.int)]

#calculating the log probability of a particular exposure, 'datum',
#on a rotated detector array, 'bowl_rot',
#to be sampled from a model, 'denisty' M
def lnprob_poisson(M,q_min,q_max,N_bin,bowl_rot,datum):
    lamb=trilinear_standin(M,q_min,q_max,bowl_rot)
    k=datum
    lnp=k*np.log(lamb)-lamb-lnfactorial(k)
    #avoiding nan inf etc.
    lnp[lamb==0]=-1000
    return np.sum(lnp)

#calculating the log probability of a particular exposure, 'datum',
#on a rotated detector array, 'bowl_rot',
#to be sampled from a model, 'denisty' M
#with solid angles accounted for
def lnprob_poisson_sa(M,q_min,q_max,N_bin,bowl_rot,datum,solid_angles='none'):
    lamb=trilinear_standin(M,q_min,q_max,bowl_rot)
    lamb*=solid_angles
    k=datum
    lnp=k*np.log(lamb)-lamb-lnfactorial(k)
    #avoiding nan inf etc.
    lnp[lamb==0]=-1000
    return np.sum(lnp)
    
########################################################################
########################################################################
#'data' is a list of the exposures
#'dect_arr' are the detector pixel q-vector coordinates
#'model_input' can be any fourier density but eventually should be noise
def emmc(data,dect_arr,model_input,q_min,q_max,N_bin,N_remodels,N_metropolis,sigma):
    print('beginning EMMC')
    t=time.time()
    
    def lnp(model,points,datum):
        return lnprob_poisson(model,q_min,q_max,N_bin,points,datum)
    
    #normalizing overall intensity
    #by simulating the expected photon count of the data
    def intensity_normalize(model): 
         t=time.time()
         print('calculating intensity scale factor')
         dpc=np.sum(data) #data photon count (take outside)
         mpc=0
         for ooo in range(len(data)):
             mpc+=np.sum(trilinear_standin(model,q_min,q_max,rand_rot(dect_arr)))
         a_intensity_scale=dpc/mpc
         print(' dpc',dpc,'mpc',mpc)
         print('a_intensity_scale',a_intensity_scale)
         model*=a_intensity_scale
         print(' ',time.time()-t,'sec')
         return model
    
    model=model_input
    intensity_normalize(model)
    models=[model]
   
    for v in range(N_remodels):
        model_val=np.zeros_like(model)
        model_weights=np.zeros_like(model)
        for d in range(np.shape(data)[0]):
            datum=data[d]
            da=np.array(rand_rot(dect_arr))
            accpt=0
            counter=0
            for l in range(N_metropolis):
                n_vec=np.array([np.random.normal(),np.random.normal(),np.random.normal()])
                n_vec/=np.sqrt(np.sum(n_vec**2))
                phi=sample_gauss_periodic(sigma)
                da_prop=rot(da,n_vec,phi)
                lnA=np.min([0,lnp(model,da_prop,datum)-lnp(model,da,datum)])
                u=np.random.uniform()
                
                '''
                if np.log(u)<=lnA:
                    da=da_prop
                    accpt+=1
                mask=np.ones_like(datum)
                (mv,mw)=trilinear_insert(da, datum, q_min, q_max, N_bin, mask)
                model_val+=np.real(mv)
                model_weights+=np.real(mw)
                model_val+=np.real(mv)
                model_weights+=np.real(mw)
                '''
                
                mask=np.ones_like(datum)
                if np.log(u)<=lnA:
                    (mv,mw)=trilinear_insert(da, datum, q_min, q_max, N_bin, mask)
                    model_val+=counter*np.real(mv)
                    model_weights+=counter*np.real(mw)
                    da=da_prop
                    accpt+=1
                    counter=0
                counter+=1
                if l==N_metropolis-1:
                    (mv,mw)=trilinear_insert(da, datum, q_min, q_max, N_bin, mask)
                    model_val+=counter*np.real(mv)
                    model_weights+=counter*np.real(mw)
                   
            if (d+1)%10==0:
                print('model',v+1,'datum ',d+1,"complete",'accpt %',100*accpt/N_metropolis)
        model_weights[model_weights==0]=1
        model=model_val/model_weights
        model[model==0]=0.000000001 #this shouldn't apply if sufficient data
        intensity_normalize(model)
        models.append(model)
    tf=time.time()-t
    if tf<120:
        print("emmc completed in",tf, "sec")
    elif tf<3600:
        print("emmc completed in",tf//60,'min',tf%60, "sec")
    else:
        print("emmc completed in",tf//3600,'hr',(tf%3600)//60, "min",(tf%3600)%60,'sec')
    return models
########################################################################
########################################################################
#'data' is a list of the exposures
#'dect_arr' are the detector pixel q-vector coordinates
#'model_input' can be any fourier density but eventually should be noise
def emmc_sa(data,dect_arr,solid_angles,model_input,q_min,q_max,N_bin,N_remodels,N_metropolis,sigma):
    print('beginning EMMC with pixel solid angles')
    t=time.time()
    
    def lnp_sa(model,points,datum):
        return lnprob_poisson_sa(model,q_min,q_max,N_bin,points,datum,solid_angles)
    
    #normalizing overall intensity
    #by simulating the expected photon count of the data
    def intensity_normalize(model): 
         t=time.time()
         print('calculating intensity scale factor')
         dpc=np.sum(data) #data photon count (take outside)
         mpc=0 #model photon count
         for ooo in range(len(data)):
             mpc+=np.sum(solid_angles*trilinear_standin(model,q_min,q_max,rand_rot(dect_arr)))
         a_intensity_scale=dpc/mpc
         print(' dpc',dpc,'mpc',mpc)
         print('intensity scale factor',a_intensity_scale)
         model*=a_intensity_scale
         print(' ',time.time()-t,'sec')
         return model
    
    model=model_input
    intensity_normalize(model)
    models=[model]
   
    for v in range(N_remodels):
        model_val=np.zeros_like(model)
        model_weights=np.zeros_like(model)
        for d in range(np.shape(data)[0]):
            datum=data[d]
            da=np.array(rand_rot(dect_arr))
            accpt=0
            counter=0
            for l in range(N_metropolis):
                n_vec=np.array([np.random.normal(),np.random.normal(),np.random.normal()])
                n_vec/=np.sqrt(np.sum(n_vec**2))
                phi=sample_gauss_periodic(sigma)
                da_prop=rot(da,n_vec,phi)
                lnA=np.min([0,lnp_sa(model,da_prop,datum)-lnp_sa(model,da,datum)])
                u=np.random.uniform()
                
                '''
                if np.log(u)<=lnA:
                    da=da_prop
                    accpt+=1
                mask=np.ones_like(datum)
                (mv,mw)=trilinear_insert(da, datum, q_min, q_max, N_bin, mask)
                model_val+=np.real(mv)
                model_weights+=np.real(mw)
                model_val+=np.real(mv)
                model_weights+=np.real(mw)
                '''
                
                mask=np.ones_like(datum)
                if np.log(u)<=lnA:
                    (mv,mw)=trilinear_insert(da, datum/solid_angles, q_min, q_max, N_bin, mask)
                    model_val+=counter*np.real(mv)
                    model_weights+=counter*np.real(mw)
                    da=da_prop
                    accpt+=1
                    counter=0
                counter+=1
                if l==N_metropolis-1:
                    (mv,mw)=trilinear_insert(da, datum/solid_angles, q_min, q_max, N_bin, mask)
                    model_val+=counter*np.real(mv)
                    model_weights+=counter*np.real(mw)
                   
            if (d+1)%10==0:
                print('model',v+1,'datum ',d+1,"complete",'accpt %',100*accpt/N_metropolis)
        model_weights[model_weights==0]=1
        model=model_val/model_weights
        model[model==0]=10**-9 #this shouldn't apply if sufficient data
        intensity_normalize(model)
        models.append(model)
    tf=time.time()-t
    if tf<120:
        print("emmc completed in",tf, "sec")
    elif tf<3600:
        print("emmc completed in",tf//60,'min',tf%60, "sec")
    else:
        print("emmc completed in",tf//3600,'hr',(tf%3600)//60, "min",(tf%3600)%60,'sec')
    return models
########################################################################
########################################################################        
                
            


#lets generate some points on a z-oriented ewald bowl to rotate around
#our bowl's radius, K
K=10
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
Nx,Ny,Nz=40,40,40
dx,dy,dz=(xmax-xmin)/(Nx-1),(ymax-ymin)/(Ny-1),(zmax-zmin)/(Nz-1)
q_max=np.array([xmax,ymax,zmax])
q_min=np.array([xmin,ymin,zmin])
N_bin=np.array([Nx,Ny,Nz])
fourier_test=np.empty([Nx,Ny,Nz])
for i in range(len(fourier_test)):
    for j in range(len(fourier_test[0])):
        for k in range(len(fourier_test[0][0])):
            (x,y,z)=(xmin+i*dx,ymin+j*dy,zmin+k*dz)
            fourier_test[i][j][k]=funct(x,y,z-1)    
#which gives a test density map named fourier_test
flat_test=np.ones(N_bin)
#give a flat density
fourier_test2=np.empty([Nx,Ny,Nz])
for i in range(len(fourier_test)):
    for j in range(len(fourier_test[0])):
        for k in range(len(fourier_test[0][0])):
            (x,y,z)=(xmin+i*dx,ymin+j*dy,zmin+k*dz)
            fourier_test2[i][j][k]=2*funct(x*2,y*2-1.75,z*2-1.5)+2*funct(x*2,y*2+1.75,z*2-1.5)+3*funct(x*2-1.5,y*2,z*2+1.5)+2*funct(x,y,z-1.5)
#gives a density with more features

#creating a random input array
model_input_random=[]
for w in range(N**3):
    model_input_random.append(np.random.uniform())
model_input_random=np.reshape(model_input_random,[N,N,N])












#model_correct=lys_intensity
model_correct=F_lys2
model_correct*=fluence*r_e**2
#model_correct/=np.max(model_correct)
#model_correct/=np.average(model_correct)
#model_correct=200*fourier_test2
#model_input=model_correct/10
#model_input=np.random.poisson(model_correct/3).astype(np.float64)+0.0001
model_input=model_input_random

#dect_arr=bowl
dect_arr=q_vecs
N_bin=40
N_bin=np.array([N_bin,N_bin,N_bin])

N_data=1000
N_remodels=2
N_steps=50
sigma=pi/2
N_buckerror=2

max_mag=max_magnitude(dect_arr)
q_max=np.array([max_mag,max_mag,max_mag])
q_min=-q_max

#fake_data=create_fake_data(model_correct,q_min,q_max,N_bin, dect_arr, N_data)
fake_data,Rs=create_fake_data_sa(model_correct, q_min, q_max, dect_arr, solid_angles,N_data,track=True)
#fake_data=create_fake_data_sa(model_correct, q_min, q_max, dect_arr,solid_angles,N_data)


#finding the best reconstruction
data_out=np.zeros_like(model_correct)
weight_out=np.zeros_like(model_correct)
for l in range(len(fake_data)):
    datum=fake_data[l]
    mask=np.ones_like(datum)
    data_out_l,weight_out_l=trilinear_insert(np.dot(dect_arr,Rs[l].T), datum, q_min, q_max, N_bin, mask)
    data_out+=np.real(data_out_l)
    weight_out+=weight_out_l
weight_out[weight_out==0]=1
best_reconstruction=data_out/weight_out

imshow_ln(best_reconstruction)



'''
#how do we use axis label?
x = np.arange(0, 10, 0.005)
y = np.exp(-x/2.) * np.sin(2*np.pi*x)

fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_xlim(0, 10)
ax.set_ylim(-1, 1)

plt.show()

bestplt=plt.subplots()
title='best reconstruction'
plt.title(title)
imshow_ln(best_reconstruction)
plt.clf()
'''

#models=emmc(fake_data,dect_arr,model_input,q_min,q_max,np.array([N,N,N]),N_remodels,N_steps,sigma)
#models=emmc_sa(fake_data,dect_arr,solid_angles,model_input,q_min,q_max,np.array([N,N,N]),N_remodels,N_steps,sigma)

#imshow_ln((models[-1]))

m=np.load("reconstruction2020-07-09.npy")
#mc_rot=error_search_ic(m,model_correct,q_min,q_max,12)[1]



#saving and documenting, edit the note!
'''
note=' full scale reconstruction with brute force error measure'

plt.clf()
now=now()
dirr="EMMC_plots/plots "+now+'/'
os.mkdir(dirr)

ti=time.time()
models=emmc_sa(fake_data,dect_arr,solid_angles,model_input,q_min,q_max,np.array([N,N,N]),N_remodels,N_steps,sigma)
tf=time.time()-ti

tie=time.time()
errors=[]
for l in range(len(models)):
    model=models[l]
    title='model '+str(l)
    plt.title(title)
    imshow_ln(model)
    plt.savefig(dirr+str(l)+' model '+now+'.pdf')
    #error=principal_axes_align(model_correct, model, q_min, q_max,return_error=True)
    if l!=0:
        if l==len(models)-1:
            error,Rot=error_search_buck(model, model_correct, q_min, q_max, N_buckerror,return_both=True)
        else:
            error=error_search_buck(model, model_correct, q_min, q_max, N_buckerror)
        errors.append(error)
    plt.clf()
tfe=time.time()-tie

title='error'
plt.title(title)
plt.plot(errors)
plt.axes.Axes.set_xticks(range(1,N_remodels+1))
plt.savefig(dirr+'error '+now+'.pdf')
plt.clf()

title='correct model rotated to reconstructed model'
plt.title(title)
#plt.xlabel('iterations')
#imshow_ln(principal_axes_align(model_correct, models[-1], q_min, q_max))
imshow_ln(rotate_density_map(model_correct, Rot, q_min, q_max))
plt.savefig(dirr+str(N_remodels+1)+' correct model rotated '+now+'.pdf')
plt.clf()

title='reconstructed model rotated to correct model'
plt.title(title)
#imshow_ln(principal_axes_align( models[-1], model_correct, q_min, q_max))
imshow_ln(rotate_density_map(models[-1], Rot.T, q_min, q_max))
plt.savefig(dirr+str(N_remodels+2)+'construct rotated to model correct '+now+'.pdf')
plt.clf()


title='best reconstruction'
plt.title(title)
imshow_ln(best_reconstruction)
plt.savefig(dirr+str(N_remodels+3)+'correct model'+now+'.pdf')
plt.clf()

text_file = open(dirr+'details '+now+'.txt', "w+")
text_file.write(str(N_data)+' data '+str(N_steps)+' steps  '+str(sigma/pi)+'pi sigma '+str(N_remodels)+ ' remodels'+' reconstucted in '+str(tf)+' secs  errors caluclated in'+str(tfe) +note)
text_file.close()

np.save(dirr+'reconstruction',models[-1])

'''




#using trlinear insert to look at a datum
data_coord=dect_arr
data_coord=rot(data_coord,[0,1,0],pi/2)
#data_coord=rot(data_coord,[1,0,0],pi/4)
#data_coord=rot(data_coord,[0,1,0],pi/2)
data_val=fake_data[0]
#data_val=trilinear_standin(fourier_test,q_min,q_max,data_coord)
#data_val=trilinear_standin(flat_test,q_min,q_max,data_coord)
one_datum=insert(data_coord, data_val, q_min, q_max, N_bin)
#imshow_x(one_datum)
imshow_collapse(one_datum)


rrr=randR()
t=time.time()
np.real(rotate3D(model_correct,[pi/4,pi/4,pi/4]))
print(t-time.time())
t=time.time()
rotate_density_map(model_correct, randR(), q_min, q_max)
print(t-time.time())








#plot_prob_phi(model_correct,model_correct,dect_arr,5000,[1,0,0],log=False,zoom=True,save=True)
#plot_prob_phi(model_correct,model_correct,dect_arr,5000,[1,1,1],log=False,zoom=True)
#plot_prob_phi(model_correct/100,model_correct,dect_arr,5000,[1,1,1])
   

#Checking to see if our proposal dist works
'''
smpls=[]
for v in range(100000):
    smpls.append(sample_gauss_periodic(pi/2.5))
    
plt.hist(smpls,bins=2*pi*np.array(range(0,100))/100-pi)
'''
#seems correct



print() 
print("done")