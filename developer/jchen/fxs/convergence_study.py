"""
Script to simulate fxs reconstruction

todo
from jugnfrau pull out smallest q - that will give the full q range

Date Created: 28 Jun 2022
Last Modified: 6 Jul 2022
Author: RAK, JC
"""

import sys
import time
import numpy as np
from reborn import utils, source, detector, dataframe, const
from reborn.target import crystal, atoms, placer
from reborn.simulate.form_factors import sphere_form_factor
from reborn.fileio.getters import FrameGetter
import pyqtgraph as pg
from reborn.viewers.qtviews import view_pad_data, scatter_plot, PADView
# import scipy.constants as const
from numpy.fft import fftn, ifftn, fftshift
from reborn.simulate.clcore import ClCore
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft

plt.close('all')

###################################################################
# Constants
###################################################################
eV = const.eV
r_e = const.r_e
NA = const.N_A
h = const.h
c = const.c
water_density = 0 #1000  # 0 for no water
rad90 = np.pi/2


#######################################################################
# Configurations
#######################################################################
pad_geometry_file = detector.jungfrau4m_geom_file #detector.epix100_geom_file
pad_binning = 4
photon_energy = 7000 * eV
detector_distance = 0.58 #2.4
pulse_energy = 0.5e-3
drop_radius = 100e-9 / 2
beam_diameter = 0.5e-6
map_resolution = 0.2e-9  # Minimum resolution for 3D density map
map_oversample = 4  # Oversampling factor for 3D density map
cell = 100e-10 #200e-10  # Unit cell size (assume P1, cubic)
pdb_file = '2W0O' #'1SS8' #'4BED'#'1SS8'  # '3IYF' '1PCQ' '2LYZ' 'BDNA25_sp.pdb'
create_bio_assembly = 1

# protein_concentration = 10 #10  # Protein concentration in mg/ml = kg/m^3
# hit_frac = 0.01  # Hit fraction
# freq = 120  # XFEL frequency
# runtime = 0.1 * 3600  # Run time in seconds
random_seed = 2022  # Seed for random number generator (choose None to make it random)
gpu_double_precision = True
gpu_group_size = 32



#########################################################################
# Program setup
#########################################################################
if random_seed is not None:
    np.random.seed(random_seed)  # Make random numbers that are reproducible

#---------------
# Make the beam
wavelength = h * c / photon_energy
beam = source.Beam(photon_energy=photon_energy, 
                   diameter_fwhm=beam_diameter, 
                   pulse_energy=pulse_energy)
beam.save_json('beam.json')
fluence = beam.photon_number_fluence

#---------------
# Make the detector
pads = detector.load_pad_geometry_list(pad_geometry_file)
for p in pads:
    p.t_vec[2] = detector_distance

pads = pads.binned(pad_binning)
q_mags = pads.q_mags(beam=beam)
solid_angles = pads.solid_angles()
polarization_factors = pads.polarization_factors(beam=beam)

#---------------
# Make the molecule (using the crystal class)
cryst = crystal.CrystalStructure(pdb_file, 
                                 spacegroup='P1', 
                                 unitcell=(cell, cell, cell, rad90, rad90, rad90), 
                                 create_bio_assembly=create_bio_assembly,
                                 tempdir='.')

dmap = crystal.CrystalDensityMap(cryst, 
                                 map_resolution, 
                                 map_oversample)
f = cryst.molecule.get_scattering_factors(beam=beam)
x = cryst.unitcell.r2x(cryst.molecule.get_centered_coordinates())
rho = dmap.place_atoms_in_map(x, f, mode='nearest')  # FIXME: replace 'nearest' with Cromer Mann densities

# Putting in surrounding water molecules
f_dens_water = atoms.xraylib_scattering_density('H2O', 
                                                water_density, 
                                                photon_energy, 
                                                approximate=True)
rho[rho != 0] -= f_dens_water * dmap.voxel_volume  # FIXME: Need a better model for solvent envelope

#---------------
# Prep the gpu for simulation
clcore = ClCore(double_precision=gpu_double_precision, group_size=gpu_group_size)

F = fftshift(fftn(rho))
# I = np.abs(F) ** 2
# rho_cell = fftshift(rho)
F_gpu = clcore.to_device(F)
q_vecs_gpu = clcore.to_device(pads.q_vecs(beam=beam))
q_mags_gpu = clcore.to_device(pads.q_mags(beam=beam))
amps_gpu = clcore.to_device(shape=pads.n_pixels, dtype=clcore.complex_t)  # This initialises zeros on the GPU of shape=shape

#---------------
# yay2



q_min = dmap.q_min
q_max = dmap.q_max


# #------------------------
# # n_shots = int(runtime * freq * hit_frac)
# n_shots = 1000 #1000 #10000
#------------------------

# #------------------------
# # protein_number_density = protein_concentration/cryst.molecule.get_molecular_weight()
# # n_proteins_per_drop = int(protein_number_density*4/3*np.pi*drop_radius**3)

# n_proteins_per_drop = 1
# #------------------------

protein_diameter = cryst.molecule.max_atomic_pair_distance  # Nominal particle size

# print('N Shots:', n_shots)
# print('Molecules per drop:', n_proteins_per_drop)
print('Particle diameter:', protein_diameter)
print('Density map grid size: (%d, %d, %d)' % tuple(dmap.shape))
print('f_dens_water: ', f_dens_water)



#########################################################################
# Visualising molecule in real space
#########################################################################
"""
print('Visualising molecule in real space')

# pyqtgraph
# pg.image(np.sum(np.fft.fftshift(np.real(rho)), axis=2))
# pg.image(np.fft.fftshift(np.real(rho)))
# yay

#-----------------
# 3D
from skimage.measure import marching_cubes
from mpl_toolkits.mplot3d import Axes3D

iso_val = 0.0
verts, faces, _, _ = marching_cubes(np.fft.fftshift(np.real(rho)), 
                                    iso_val, 
                                    spacing=(0.1, 0.1, 0.1))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(verts[:, 0], 
                verts[:,1], 
                faces, 
                verts[:, 2], 
                lw=1)
plt.show(block=False)


#-----------------
# Projection plots
Nx, Ny, Nz = rho.shape
Nx_cent = int(np.round(Nx/2))
Ny_cent = int(np.round(Ny/2))
Nz_cent = int(np.round(Nz/2))

def show_projection(disp_map, disp_str):
    """
    Projection
    """
    fig = plt.figure()
    ax = fig.add_subplot(131)
    im = ax.imshow(np.sum(disp_map, axis=0), origin='lower')
    fig.colorbar(im, shrink=0.5, ax=ax)
    # ax.set_xlim([60,150])
    # ax.set_ylim([60,150])
    ax.set_title('np.sum(axis=0)')
    ax = fig.add_subplot(132)
    im = ax.imshow(np.sum(disp_map, axis=1), origin='lower')
    fig.colorbar(im, shrink=0.5, ax=ax)
    # ax.set_xlim([60,150])
    # ax.set_ylim([60,150])
    ax.set_title('np.sum(axis=1)')
    ax = fig.add_subplot(133)
    im = ax.imshow(np.sum(disp_map, axis=2), origin='lower')
    fig.colorbar(im, shrink=0.5, ax=ax)
    # ax.set_xlim([60,150])
    # ax.set_ylim([60,150])
    ax.set_title('np.sum(axis=2)')

    plt.suptitle(disp_str)
    plt.tight_layout()
    plt.show(block=False)


disp_map = np.fft.fftshift(np.real(rho))
disp_str = 'Projection of molecule in real space in the 3 directions'
show_projection(disp_map, disp_str)

#----------------------


"""
# yay1
################################################################################
# FXS Convergence study
################################################################################
print('Begin convergence study')

s = 2  # Oversampling
d = 8e-10 #8e-10  # Autocorrelation ring resolution
q = 2 * np.pi / d  # q magnitude of autocorrelation ring (2pi/d)

n_phi = int(2 * np.pi * s * protein_diameter / d)  # Num angular bins in ring
n_phi += (n_phi % 2)
print('Nphi (num angular bins):', n_phi)

dphi = 2*np.pi / n_phi  # Angular step in phi
phi = np.arange(n_phi) * dphi

dtheta = wavelength / s / protein_diameter  # Angular step in theta
theta = 2 * np.arcsin(q * wavelength / 4 / np.pi)
print(f'2 theta (deg): {2 * theta * 180 / np.pi:.3f}')

st = np.sin(theta)
sa = st * dphi * dtheta  # Ring bin solid angle
q_ring = 2 * np.pi / wavelength * np.vstack(
    [st * np.cos(phi), st * np.sin(phi), (1 - np.cos(theta)) * np.ones(n_phi)]).T.copy()
q_ring_gpu = clcore.to_device(q_ring)
a_ring = clcore.to_device(shape=(n_phi,), dtype=clcore.complex_t) # This initialises zeros on the GPU of shape=shape



def autocorr_do(n_shots, n_proteins_per_drop):

    # acf_sum_noisy = np.zeros(n_phi)
    acf_sum = np.zeros(n_phi)

    for i in range(n_shots):
        if ((i + 1) % 100) == 0:
            print(i + 1)
        
        p_vecs = placer.particles_in_a_sphere(sphere_diameter=drop_radius * 2, n_particles=n_proteins_per_drop, particle_diameter=protein_diameter)
        
        for p in range(n_proteins_per_drop):
            add = 1  # Default: add amplitudes of protein diffraction
            if p == 0:  # For the first molecule, do not add amplitudes.  Overwrite the GPU memory buffer instead.
                add = 0
            R = Rotation.random().as_matrix()
            U = p_vecs[p, :]
            clcore.mesh_interpolation(F_gpu, 
                                      q_ring_gpu, 
                                      N=dmap.shape, 
                                      q_min=q_min, 
                                      q_max=q_max, 
                                      R=R, 
                                      U=U, 
                                      a=a_ring,
                                      add=add) # This is where a_ring gets modified
        I_ring = np.abs(a_ring.get()) ** 2
        I_ring *= sa * r_e ** 2 * fluence

        # I_ring_noisy = np.random.poisson(I_ring).astype(np.float64)

        I_ring -= np.mean(I_ring)
        acf_sum += np.real(ifft(np.abs(fft(I_ring))**2))

        # I_ring_noisy -= np.mean(I_ring_noisy)
        # acf_sum_noisy += np.real(ifft(np.abs(fft(I_ring_noisy))**2))


    # Normalizations
    acf_sum /= n_shots
    acf_sum -= np.median(acf_sum)


    return acf_sum



"""
n_proteins_per_drop1 = 1
n_shots1 = 10000
acf_sum1 = autocorr_do(n_shots=n_shots1, n_proteins_per_drop=n_proteins_per_drop1)


print('Doing the convergence study')
n_proteins_per_drop2 = 10
n_shots2 = 100
acf_sum2 = autocorr_do(n_shots=n_shots2, n_proteins_per_drop=n_proteins_per_drop2)

n_proteins_per_drop3 = 10
n_shots3 = 1000
acf_sum3 = autocorr_do(n_shots=n_shots3, n_proteins_per_drop=n_proteins_per_drop3)

n_proteins_per_drop4 = 10
n_shots4 = 10000
acf_sum4 = autocorr_do(n_shots=n_shots4, n_proteins_per_drop=n_proteins_per_drop4)

# m = int(n_phi / 2)
m = n_phi
phis = phi[1:m] * 180 / np.pi # Phi-s for plotting

plt.figure()
plt.plot(phis, acf_sum2[1:m], '-', color='tab:orange', label=f'num prot per drop: {n_proteins_per_drop2:d}, num shots: {n_shots2:d}')
plt.plot(phis, acf_sum3[1:m], '-', color='tab:red', label=f'num prot per drop: {n_proteins_per_drop3:d}, num shots: {n_shots3:d}')
plt.plot(phis, acf_sum4[1:m], '-', color='tab:green', label=f'num prot per drop: {n_proteins_per_drop4:d}, num shots: {n_shots4:d}')
plt.plot(phis, acf_sum1[1:m], '--', lw=2, color='black', label=f'num prot per drop: {n_proteins_per_drop1:d}, num shots: {n_shots1:d}')
# acf_sum_noisy += np.mean(acf_sum[1:m]) - np.mean(acf_sum_noisy[1:m])
# plt.plot(phi[1:m] * 180 / np.pi, acf_sum_noisy[1:m], '.r')
plt.xlabel(r'$\Delta \phi$ (degrees)')
plt.ylabel(r'$C(q, q, \Delta\phi)$')
plt.grid()
plt.legend()
plt.show(block=False)
"""

#--------------------------
# Convergence calculations

print('calculating reference')
n_shots_ref = 500
acf_avg_ref = autocorr_do(n_shots=n_shots_ref, n_proteins_per_drop=1)


print('calculating convergence errors')
n_shots_tot = 500
protein_concentration = np.array([10, 20, 30]) #np.array([10, 100, 1000]) #10  # Protein concentration in mg/ml = kg/m^3
n_shot_mod = 100 # Calculate error every this many shots


protein_number_density = protein_concentration/cryst.molecule.get_molecular_weight()
num_proteins_per_drop_vec = np.round(protein_number_density*4/3*np.pi*drop_radius**3).astype(int)

# #------------
# # Insert a 1 at the start
# index = 0
# value = 1
# num_proteins_per_drop_vec = np.insert(num_proteins_per_drop_vec, index, value)
# print(num_proteins_per_drop_vec)
# #------------

protein_concentration_len = len(num_proteins_per_drop_vec)

n_errors = int(n_shots_tot / n_shot_mod)
errors = np.zeros((n_errors, protein_concentration_len))

for i_proteins_per_drop in range(protein_concentration_len):
    print(f'Doing {num_proteins_per_drop_vec[i_proteins_per_drop]} proteins')
    n_proteins_per_drop = num_proteins_per_drop_vec[i_proteins_per_drop]
    

    acf_sum = np.zeros(n_phi)
    

    ind = 0 # error index
    for i_shots in range(n_shots_tot):
        
        p_vecs = placer.particles_in_a_sphere(sphere_diameter=drop_radius * 2, n_particles=n_proteins_per_drop, particle_diameter=protein_diameter)
        

        for p in range(n_proteins_per_drop):
            add = 1  # Default: add amplitudes of protein diffraction
            if p == 0:  # For the first molecule, do not add amplitudes.  Overwrite the GPU memory buffer instead.
                add = 0
            R = Rotation.random().as_matrix()
            U = p_vecs[p, :]
            clcore.mesh_interpolation(F_gpu, 
                                      q_ring_gpu, 
                                      N=dmap.shape, 
                                      q_min=q_min, 
                                      q_max=q_max, 
                                      R=R, 
                                      U=U, 
                                      a=a_ring,
                                      add=add) # This is where a_ring gets modified
        I_ring = np.abs(a_ring.get()) ** 2
        I_ring *= sa * r_e ** 2 * fluence

        # I_ring_noisy = np.random.poisson(I_ring).astype(np.float64)

        I_ring -= np.mean(I_ring)
        acf_sum += np.real(ifft(np.abs(fft(I_ring))**2))

        # I_ring_noisy -= np.mean(I_ring_noisy)
        # acf_sum_noisy += np.real(ifft(np.abs(fft(I_ring_noisy))**2))


        # Calculate the difference every n_shot_mod shots
        if ((i_shots + 1) % n_shot_mod) == 0:
            print(i_shots + 1)

            # Normalizations
            acf_avg_curr = acf_sum/(i_shots+1) # plus 1 because Python starts counting at 0
            acf_avg_curr -= np.median(acf_avg_curr)

            # plt.figure()
            # plt.plot(acf_avg_curr[10:-10])
            # plt.plot(acf_avg_ref[10:-10], 'k--')
            # plt.grid()
            # plt.show(block=False)
            # yay

            errors[ind, i_proteins_per_drop] = (np.sum((acf_avg_curr[10:-10] - acf_avg_ref[10:-10])**2))
            ind += 1

errors = np.sqrt(errors) / np.sqrt(np.sum(acf_avg_ref[10:-10]**2))


# plt.figure()
# for i_proteins_per_drop in range(protein_concentration_len):
#     plt.plot(np.arange(1,len(errors)+1)*100, (errors[:,i_proteins_per_drop]), '-', label=f'protein concentration: {protein_concentration[i_proteins_per_drop]} mg/ml')
# plt.xlabel('Num shots')
# plt.ylabel('Root mean squared error')
# # plt.ylim([0,50])
# plt.grid()
# plt.legend()
# plt.show(block=False)


np.savez(file=f'autocorr_dat1',
         n_shots_ref=n_shots_ref,
         n_shots_tot=n_shots_tot,
         protein_concentration=protein_concentration,
         n_shot_mod=n_shot_mod,
         errors=errors,
         water_density=water_density)






"""
# Load code
dat = np.load('autocorr_dat1.npz')

n_shots_ref = dat['n_shots_ref']
n_shots_tot = dat['n_shots_tot']
protein_concentration = dat['protein_concentration']
n_shot_mod = dat['n_shot_mod']
errors = dat['errors']
water_density = dat['water_density']

protein_concentration_len = len(protein_concentration)


plt.figure()
for i_proteins_per_drop in range(protein_concentration_len):
    plt.plot(np.arange(1,len(errors)+1)*100, (errors[:,i_proteins_per_drop]), '-', label=f'protein concentration: {protein_concentration[i_proteins_per_drop]} mg/ml')
plt.xlabel('Num shots')
plt.ylabel('Root mean squared error')
# plt.ylim([0,50])
plt.grid()
plt.legend()
plt.show(block=False)
"""





