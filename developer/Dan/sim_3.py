import time
import numpy as np
from scipy import constants as const
from scipy.spatial.transform import Rotation
import scipy.optimize
import matplotlib.pyplot as plt
import reborn
import reborn.target.crystal as crystal
import reborn.simulate.clcore as core
import reborn.simulate.solutions as solutions
import reborn.utils as utils
from reborn.utils import random_unit_vector
from reborn import simulate
from reborn.utils import trilinear_insert
from reborn.simulate.form_factors import sphere_form_factor
from reborn.simulate import solutions

# %%
eV = const.value('electron volt')
r_e = const.value('classical electron radius')
pdb_id = '2LYZ'
h = const.h
c = const.c
photon_energy = 9000*eV
pulse_energy = 1.1326e-1
pixel_size = 100e-6
distance = 0.25
nff = 2001
nss = 2001
resolution = 8e-10
s = 2
sph_rad = 500e-10
fft = 1

# %%
simcore = core.ClCore(group_size=32, double_precision=False)

# %%
beam = reborn.source.Beam(photon_energy=photon_energy, pulse_energy = pulse_energy)
pad = reborn.detector.PADGeometry(shape=(nss, nff), pixel_size=pixel_size, distance=distance)
q_vecs = pad.q_vecs(beam=beam)
q_mags = pad.q_mags(beam=beam)
n_pixels = q_vecs.shape[0]
wl = beam.wavelength
fluence = beam.photon_number_fluence
sa = pad.solid_angles()
pf = pad.polarization_factors(beam=beam)

print('fluence =', fluence)
#print('wl =', wl)
print('fft =', fft)

# %%
cryst = crystal.CrystalStructure(pdb_id, tight_packing=True)
r_vecs = cryst.molecule.coordinates  # These are atomic coordinates (Nx3 array)

# %% Scattering factors
f = cryst.molecule.get_scattering_factors(beam=beam)

atomic_numbers = cryst.molecule.atomic_numbers
n_atoms = r_vecs.shape[0]

# %%
# "Diameter" of the molecule
diam = 0
for i in range(r_vecs.shape[0]):
    for j in range(r_vecs.shape[0]):
        d = np.sqrt((r_vecs[i,0] - r_vecs[j,0])**2 
            + (r_vecs[i,1] - r_vecs[j,1])**2 + (r_vecs[i,2] - r_vecs[j,2])**2)
#        av_d += d    
        if d > diam: 
            diam = d
            k = i
            l = j
diam_c = (r_vecs[k,:] + r_vecs[l,:])/2
mol_rad = diam/2
r_vecs -= diam_c/2 # Center the molecule
#os = 4
#dr = resolution / s / os
#r_vecs = np.rint(r_vecs/dr)*dr

# %% Scattering factors corrections
#at_num = np.array([6,7,8,16])   
#at_rad = np.array([70e-12,65e-12,60e-12,100e-12])  
#uniq_z = np.unique(atomic_numbers)
#for z in uniq_z:
#    radius = np.squeeze(at_rad[np.where(at_num == z)])
#    df = (4*np.pi*radius**3)/3*sph_dens
#    f1[np.where(atomic_numbers == z)] -= df    

# %%
q = s * np.pi / resolution
n = int(q * sph_rad / np.pi - 0.5) # Minimum ring number

# %% n-th minimum ring
def fcl(x):
    return np.tan(x) - x
root_x = scipy.optimize.brentq(fcl, ((n-1)+0.5001)*np.pi,(n+0.4999)*np.pi)
q = root_x/sph_rad 
#resolution = 2 * np.pi / q
theta = 2 * np.arcsin(q * wl / (4 * np.pi)) 
theta_max = np.arctan(nss/2*pixel_size/distance)
print('theta =', theta) 
#print('theta_max =', theta_max) 
print('resolution =', resolution * 1e10, 'A') 

# %% 
# q-vectors on the ring
n = int(2 * np.pi * diam / resolution) + 1 #Number of samples on the ring
ns = 40
nsq = 40
nsp = 1
print('N_phi =', 2*n)
print('nq =', nsq)
dphi = np.pi/(n-1)
#dq = 2 * np.pi * np.cos(theta) / s / nsq / diam
dq = np.pi / ns / sph_rad
wlc = 2*np.pi/wl
vecs = np.empty((n, 3))
q_vecs = np.empty((n, 3))
qr_vecs = np.empty((n*nsp*nsq, 3))
mx = 0
for m in range(0, n):
    phi = m * dphi
    vecs [m,0] = np.sin(theta) * np.cos(phi)
    vecs [m,1] = np.sin(theta) * np.sin(phi)
    vecs [m,2] = np.cos(theta)
    q_vecs [m,0] = q * np.cos(phi)
    q_vecs [m,1] = q * np.sin(phi)
    q_vecs [m,2] = 0
    for mp in range (0, nsp):    
        for mq in range (0, nsq):    
            phi = m * dphi + ((1-nsp)/2 + mp) * dphi/nsp
            qr_vecs [mx,0] = (q + ((1-nsq)/2 + mq) * dq) * np.cos(phi)
            qr_vecs [mx,1] = (q + ((1-nsq)/2 + mq) * dq) * np.sin(phi)
            qr_vecs [mx,2] = 0
            mx += 1

# %% Polarization factors and solid angles
pf_r = 1 - vecs[:,0]**2   
sa_r = np.ones(n) * wl**2 / (s**2 * nsp * ns * diam**2)
print()    
#ind = pad.vectors_to_indices(vecs, insist_in_pad=True, round=True)
#ind = np.array(ind[1] + nss * ind[0], dtype = np.int32)
#q_vecs_r = []
#pf_r = []
#A_sph_r = []
#for k in ind:
#    q_vecs_r.append(q_vecs[k,:])
#    pf_r.append(pf[k])
#    A_sph_r.append(A_sph[k])   
#q_vecs_r = np.array(q_vecs_r)
#pf_r = np.array(pf_r) 
#A_sph_r = np.array(A_sph_r) 

#sa_r = np.ones(n) * wl * resolution * np.sin(theta/2) / (2 * diam**2) 
#f = np.ones(r_vecs.shape[0]) * 7.0 * np.exp(-10.7 * (np.sin(theta/2)/(wl*1e10))**2)
#f = np.ones(r_vecs.shape[0]) * 7 * np.exp(-10.7 /(2*resolution*1e10)**2)



# %% Droplet intensity
q_mags = utils.vec_mag(qr_vecs)
f_sph = simulate.form_factors.sphere_form_factor(radius=sph_rad, q_mags=q_mags)
f_water = solutions.get_water_profile(q=q_mags, temperature=300)
f_water_0 = solutions.get_water_profile(q=0, temperature=300)
n_water = solutions.water_number_density()
A_sph = f_sph * f_water * n_water
Id_sph = np.array([part.sum() for part in np.split(np.abs(A_sph)**2, n)])

# %%
# Grid inside the sphere
N = int(sph_rad / diam - 0.5)
grid = []
for x in range(-N,N+1):
    for y in range(-N,N+1):
        for z in range(-N,N+1):
            if x**2+y**2+z**2 <= N**2:
                 vec = np.array([x,y,z]) * diam
                 grid.append(vec)
NR = len(grid) # Number of points on the grid 

# %%
# Water molecules map
dr_w = n_water**(-1/3)
N = int(2 * np.rint(diam / dr_w / 2))
off = - (N-1)/2
rw_vecs = []
for x in range(0,N):
    for y in range(0,N):
        for z in range(0,N):
            r = random_unit_vector() * 0.0            
            vec = (np.array([x,y,z]) + off + r) * dr_w
            rw_vecs.append(vec)
rw_vecs = np.array(rw_vecs)            
fw = np.full(rw_vecs.shape[0],f_water_0)

# %%
q_max = s * np.pi / resolution
NS = 600  # Number of mesh samples
print('NS =', NS)
if fft == 0:
# %% Density map
    n_bin = np.array([NS,NS,NS])
    os = 4
    dr = resolution / s / os
    q_max = os * s * np.pi / resolution
    cr = (NS-1)*dr
    rd_vecs = np.rint(r_vecs/dr)*dr
    x_min = np.array([0,0,0])
    x_max = np.array([cr,cr,cr])
    
    NW = int(2 * np.rint(dr_w * N / dr / 2)) # Background map 
    f_bg = - 2 * N**3 / NW**3 * f_water_0
    rbg_vecs = []
    off = - (NW-1)/2
    for x in range(0,NW):
        for y in range(0,NW):
            for z in range(0,NW):
                vec = (np.array([x,y,z]) + off) * dr
                rbg_vecs.append(vec)
    rbg_vecs = np.array(rbg_vecs)  
    fbg = np.full(rbg_vecs.shape[0],f_bg)
    
    r_map = np.ascontiguousarray(np.concatenate((rd_vecs,rw_vecs,rbg_vecs),axis=0))
    f_map = np.ascontiguousarray(np.concatenate((np.abs(f),fw,fbg)))
    mask=np.full(len(f_map), True, dtype=bool)
    au_map, weights = trilinear_insert(r_map, data_val=f_map, x_min=x_min, x_max=x_max, n_bin=n_bin, mask=mask, boundary_mode='periodic')
# %% FFT    
    a_map_dev = simcore.to_device(np.fft.fftshift(np.fft.fftn(au_map)), dtype=simcore.complex_t) 
else:
    a_map_dev = simcore.to_device(shape=(NS ** 3,), dtype=simcore.complex_t)
    simcore.phase_factor_mesh(r_vecs, f, N=NS, q_min=-q_max, q_max=q_max, a=a_map_dev)
    
# %% Pre-allocation of GPU arrays
q_dev = simcore.to_device(q_vecs, dtype=simcore.real_t)
r_dev = simcore.to_device(r_vecs, dtype=simcore.real_t)
f_dev = simcore.to_device(f, dtype=simcore.complex_t)

print('1 - done')

N = 1 # Number of molecules
M = 1000000 # Number of snapshots

rng = np.random.default_rng()
#nr = int(sph_rad / mol_rad / s) + 1
#print('nr = ', nr)

#nf = np.empty([M,n*ns**2])
nfs = np.empty([M,n])
#nfp = np.empty([M,n*ns**2])
#I = np.zeros(n*ns**2)
#cf = np.zeros([ns**2, n*ns**2])
#isq = np.zeros(n*ns**2)
#Ip = np.zeros(n*ns**2)
#cfp = np.zeros([ns**2, n*ns**2])
#isqp = np.zeros(n*ns**2)
for k in range(0, M):
    a_dev = simcore.to_device(shape=(q_dev.shape[0]), dtype=simcore.complex_t)
#    a = np.random.rand()
#    rad = sph_rad
#    f_sph = simulate.form_factors.sphere_form_factor(radius=rad, q_mags=q_mags)
#    A_sph = f_sph * f_water * n_water
    for i in range(0, N):
        gn = np.random.randint(NR)
        U = grid[gn]
        R = Rotation.random().as_matrix()
        simcore.mesh_interpolation(a_map_dev, q_dev, N=NS, q_min=-q_max, q_max=q_max, a=a_dev, R=R, U=U, add=True)
#        simcore.phase_factor_qrf(q_dev, r_dev, f=f_dev, a=a_dev, R=R)
    A = a_dev.get()
    Ids = np.abs(A)**2 * nsq
#    Ids += Id_sph
#    Ids = np.array([part.sum() for part in np.split(np.abs(A_sph)**2, n)])
#    Ids = np.array([part.sum() for part in np.split(np.abs(A)**2, n)])
#    Ids = np.array([part.sum() for part in np.split(np.abs(A)**2+np.abs(A_sph)**2, n)])
#    Idsp = rng.poisson(Ids).astype(np.double)
    Ids *= r_e**2 * fluence * sa_r * pf_r
#    for k in range(0, ns**2):
#        cf[k,:] = +Ids[k] * Ids
#        cfp[k,:] = +Idsp[k] * Idsp    
    
    
    
#    nf[k,:] = Id
    nfs[k,:] = Ids
#    nfp[k,:] = rng.poisson(Id).astype(np.double)

#    I += Id
#    a = Id[0]
 #   cf += Id * a
#    isq += Id**2 * a**2
#    Idp = rng.poisson(Id).astype(np.double)
#    Ip += Idp
#    ap = Idp[0]
#    cfp += Idp * ap
#    isqp += (Idp * ap)**2
    
print('2 - done') 

nfp = rng.poisson(nfs).astype(np.double)

#cf /= M
#cfp /= M
 
#np.save('ints.npy', nf)
np.save('intss.npy', nfs)
#np.save('cf.npy', cf)
#np.save('cfp.npy', cfp)

I = np.mean(nfs,axis=0)
nfs -= I
Ip = np.mean(nfp,axis=0)
nfp -= Ip
for k in range(0,M):
    nfs[k,:] *= nfs[k,0] 
    nfp[k,:] *= nfp[k,0] 
cfs = np.mean(nfs,axis=0)
cfp = np.mean(nfp,axis=0)
cfeb = np.sqrt(np.mean(nfp**2,axis=0)-cfp**2)/np.sqrt(M)


file1 = open('qwe.txt', 'w')
print('<n(q)> =', np.mean(I)*1e3,'e-3')
#print('<n(q)_p> =', np.mean(Ip)*1e3,'e-3', file = file1)

n1 = int(0.1*n)
n2 = int(0.9*n)
rms_cf = 0
rms_var = 0
dsnr_num = 0
avcfs = 0
for k in range(n1, n2):
    avcfs += cfs[k] 
avcfs /= n2-n1
#avcf1 = np.mean(cf)
for k in range(n1, n2): 
#    rms_cf += (cfs[k]-avcfs)**2
    rms_cf += (cfs[k])**2
    rms_var += (cfp[k]-cfs[k])**2
    dsnr_num += ((cfp[k]-cfs[k])*cfeb[k])**2
av_snrp = np.sqrt(rms_cf)/np.sqrt(rms_var)
d_snr = av_snrp * np.sqrt(dsnr_num) / rms_var
print('<snr> =', av_snrp)
print('<d_snr> =', d_snr)



cfs /= np.mean(I)**2
cfp /= np.mean(Ip)**2
cfeb /= np.mean(Ip)**2

x = np.arange(n)
f = plt.figure(figsize=plt.figaspect(0.4))
#plt.subplot(121)
plt.errorbar(x*180/(n-1), cfp, yerr = cfeb)
plt.plot(x*180/(n-1), cfs, color="k")
plt.xticks(np.arange(181, step=90))
plt.ylim(top=cfs[0])
#plt.title('IntCorrs')
#plt.subplot(122)
#plt.plot(x*180/(n-1), snrp, 'r.')
#plt.plot(x*180/(n-1), snr, color="k")
#plt.xticks(np.arange(181, step=90))
#plt.title('SNR')
f.savefig("corrs_snr.png", bbox_inches='tight')
plt.show()
