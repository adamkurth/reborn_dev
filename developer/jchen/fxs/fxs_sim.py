"""
Script to simulate fxs reconstruction

Date Created: 2021
Last Modified: 25 Jun 2022
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


#------------------------
sel = 0
#------------------------

"""
- Start with two particles and cmp one particle see if they agree
- and then vary concentration 
- look in proposal 
"""

###################################################################
# Constants
###################################################################
eV = const.eV
r_e = const.r_e
NA = const.N_A
h = const.h
c = const.c
water_density = 1000
rad90 = np.pi/2


# Change epix100_geom_file to jungerfau - D
# simulate 1e6 shots
# 5 different samples - do this for each to find best sample
# 50% glecerol - we will use most of time - so put 50% density of glecerol in place in stead o water
# make % glerol a parameter
# wavelength - may be 7-10kev - depend on background/


"""
Make google doc
For each sample with detector, wavlength - what diffraction looks like
- Are central speckles missing (enough for phase retrieval)
- 70nm sindbus - just do 70nm sphere (maybe get pdb)
- water droplet 100-200nm water drop - are fringes separated well enough so we can correlate zeros
- single protein no noise

# add ifpoisson
# add if nppd
# see projection of molecule
# Make biological correct thing - PSI trimer, DNA 25 base pair, DNA gold balls on end, Groel, Groel Pt, apoferrin, ribosome, sinbus, 
# Git hash for each - just copy parameter for 
# Apo fretin groel



Big pic
- get bio working - PSI trimer, DNA 25 base pair, DNA gold balls on end, Groel, Groel Pt, apoferrin, ribosome, sinbus, 
- convergence - see N --> 1
- add noise
- give roberto patterns script maker

Longer term
- simulation of beam - Gaussian beam, jitter (droplet, x-ray beam same size!)

"""


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

protein_concentration = 10  # Protein concentration in mg/ml = kg/m^3
hit_frac = 0.01  # Hit fraction
freq = 120  # XFEL frequency
runtime = 0.1 * 3600  # Run time in seconds
random_seed = 2022  # Seed for random number generator (choose None to make it random)
gpu_double_precision = True
gpu_group_size = 32

#########################################################################
# Derived parameters
#########################################################################
if random_seed is not None:
    np.random.seed(random_seed)  # Make random numbers that are reproducible

#------------------------
# n_shots = int(runtime * freq * hit_frac)
n_shots = 10000 #10000
#------------------------


wavelength = h * c / photon_energy
beam = source.Beam(photon_energy=photon_energy, 
                   diameter_fwhm=beam_diameter, 
                   pulse_energy=pulse_energy)
beam.save_json('beam.json')
fluence = beam.photon_number_fluence


#------------------------ 
f_dens_water = atoms.xraylib_scattering_density('H2O', 
                                                water_density, 
                                                photon_energy, 
                                                approximate=True)
# f_dens_water = 0
#------------------------

pads = detector.load_pad_geometry_list(pad_geometry_file)
for p in pads:
    p.t_vec[2] = detector_distance

pads = pads.binned(pad_binning)
q_mags = pads.q_mags(beam=beam)
solid_angles = pads.solid_angles()
polarization_factors = pads.polarization_factors(beam=beam)
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


#----------------------
print('Visualising molecule in real space')

# pyqtgraph
# pg.image(np.sum(np.fft.fftshift(np.real(rho)), axis=2))
# pg.image(np.fft.fftshift(np.real(rho)))
# yay

#-----------------
# 3D
from skimage.measure import marching_cubes
from mpl_toolkits.mplot3d import Axes3D

iso_val=0.0
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
Nx_cent = int(np.floor(Nx/2))
Ny_cent = int(np.floor(Ny/2))
Nz_cent = int(np.floor(Nz/2))

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



rho[rho != 0] -= f_dens_water * dmap.voxel_volume  # FIXME: Need a better model for solvent envelope
F = fftshift(fftn(rho))
I = np.abs(F) ** 2
rho_cell = fftshift(rho)
clcore = ClCore(double_precision=gpu_double_precision, group_size=gpu_group_size)
F_gpu = clcore.to_device(F)
q_vecs_gpu = clcore.to_device(pads.q_vecs(beam=beam))
q_mags_gpu = clcore.to_device(pads.q_mags(beam=beam))
amps_gpu = clcore.to_device(shape=pads.n_pixels, dtype=clcore.complex_t)  # This initialises zeros on the GPU of shape=shape

# yay2

q_min = dmap.q_min
q_max = dmap.q_max
protein_number_density = protein_concentration/cryst.molecule.get_molecular_weight()

#------------------------
# n_proteins_per_drop = int(protein_number_density*4/3*np.pi*drop_radius**3)
n_proteins_per_drop = 1
#------------------------

protein_diameter = cryst.molecule.max_atomic_pair_distance  # Nominal particle size

print('N Shots:', n_shots)
print('Molecules per drop:', n_proteins_per_drop)
print('Particle diameter:', protein_diameter)
print('Density map grid size: (%d, %d, %d)' % tuple(dmap.shape))

# yay1
###########################################################################
# Water droplet with protein, 2D PAD simulation
##########################################################################
class DropletGetter(FrameGetter):
    def __init__(self, n_frames, drop_radius, n_proteins_per_drop):
        super().__init__()
        self.n_frames = n_frames #1e1 #np.inf
        self.drop_radius = drop_radius
        self.n_proteins_per_drop = n_proteins_per_drop

    def get_data(self, frame_number=0):
        # dd = drop_radius*2 #+ (np.random.rand()-0.5)*drop_radius/5
        # nppd = 1 #int(protein_number_density*4/3*np.pi*(dd/2)**3)

        dd = drop_radius*2
        nppd = self.n_proteins_per_drop

        p_vecs = placer.particles_in_a_sphere(sphere_diameter=dd, 
                                              n_particles=nppd, 
                                              particle_diameter=protein_diameter)
        add = False
        for p in range(nppd):
            R = Rotation.random().as_matrix()
            U = p_vecs[p, :]

            # Modify amps_gpu
            clcore.mesh_interpolation(F_gpu,
                                      q_vecs_gpu,
                                      N=dmap.shape,
                                      q_min=q_min,
                                      q_max=q_max,
                                      R=R,
                                      U=U,
                                      a=amps_gpu,
                                      add=add)
            add = True
        
        # # clcore.sphere_form_factor(r=dd/2, q=q_mags_gpu, a=amps_gpu, add=add)
        # sff = f_dens_water*sphere_form_factor(radius=dd/2, q_mags=q_mags)
        # I = np.abs(amps_gpu.get()+sff)**2 * r_e**2 * solid_angles * polarization_factors * fluence

        # Calculate the diffracted intensity of the molecule
        I = np.abs(amps_gpu.get())** 2 * r_e**2 * solid_angles * polarization_factors * fluence
        
        # Sample Poisson noise
        # I = np.random.poisson(I)

        df = dataframe.DataFrame(pad_geometry=pads, beam=beam, raw_data=I)
        return df

fg = DropletGetter(n_frames=1e1, 
                   drop_radius=drop_radius, 
                   n_proteins_per_drop=n_proteins_per_drop)
pv = PADView(frame_getter=fg, hold_levels=True)
pv.start()

# sys.exit()

yay

################################################################################
# FXS Simulation
################################################################################


# from jugnfrau pull out smallest q * - that's got full range

if sel == 0:
    print('doing sel 0 (original)')

    s = 2  # Oversampling
    d = 8e-10 #8e-10  # Autocorrelation ring resolution
    q = 2 * np.pi / d  # q magnitude of autocorrelation ring (2pi/d)
    n_phi = int(2 * np.pi * s * protein_diameter / d)  # Num angular bins in ring
    n_phi += (n_phi % 2)
    print('Nphi (num angular bins):', n_phi)

    dphi = 2 * np.pi / n_phi  # Angular step in phi
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

    acf_sum_noisy = np.zeros(n_phi)
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



    plt.figure()
    plt.plot(acf_sum_noisy)
    plt.show(block=False)



    # m = int(n_phi / 2)
    m = n_phi

    plt.figure(2)
    plt.plot(phi[1:m] * 180 / np.pi, acf_sum[1:m], '-k', label='noiseless')
    # acf_sum_noisy += np.mean(acf_sum[1:m]) - np.mean(acf_sum_noisy[1:m])
    # plt.plot(phi[1:m] * 180 / np.pi, acf_sum_noisy[1:m], '.r')
    plt.xlabel(r'$\Delta \phi$ (degrees)')
    plt.ylabel(r'$C(q, q, \Delta\phi)$')
    plt.grid()
    plt.legend()
    plt.show(block=False)
    # view_pad_data(pad_data=np.log10(I_sphere + 1), pad_geometry=pads, show=True)
    # view_pad_data(pad_data=np.random.poisson(np.log10(I_prot + 1)), pad_geometry=pads, show=True)

    # yay4




    if 0:
        #------------------
        # np.savez(file='nshot=1e4_water=n_d=15e-10_n-proteins-per-drop=1',
        #          phi=phi,
        #          acf_sum=acf_sum)

        dat = np.load('nshot=1e4_water=n_d=8e-10_n-proteins-per-drop=1.npz')

        plt.figure(3)
        plt.plot(phi[1:m] * 180 / np.pi, acf_sum[1:m], '-', lw=2, label='with water')
        plt.plot(phi[1:m] * 180 / np.pi, dat['acf_sum'][1:m], '-', lw=2, label='without water')
        plt.xlabel(r'$\Delta \phi$ (degrees)', fontsize=12)
        plt.ylabel(r'$C(q, q, \Delta\phi)$', fontsize=12)
        plt.title(f'Ring autocorrelation, no Poisson noise, pdb={pdb_file}, n_shots={n_shots}, resolution={d}, n_proteins_per_drop={n_proteins_per_drop}')
        plt.grid()
        plt.ylim([-0.015, 0.06])
        plt.legend()
        plt.show(block=False)

        #------------------




    if 0:
        fig = plt.figure()
        fig.add_subplot(2, 3, 1)
        I = np.abs(I).astype(np.float64)
        rho = np.abs(rho).astype(np.float64)
        dispim = np.log10(I[int(dmap.shape[0] / 2), :, :] + 10)
        plt.imshow(dispim, interpolation='nearest', cmap='gray')
        fig.add_subplot(2, 3, 4)
        dispim = np.sum(rho, axis=0)
        plt.imshow(dispim, interpolation='nearest', cmap='gray')
        fig.add_subplot(2, 3, 2)
        dispim = np.log10(I[:, int(dmap.shape[1] / 2), :] + 10)
        plt.imshow(dispim, interpolation='nearest', cmap='gray')
        fig.add_subplot(2, 3, 5)
        dispim = np.sum(rho, axis=1)
        plt.imshow(dispim, interpolation='nearest', cmap='gray')
        fig.add_subplot(2, 3, 3)
        dispim = np.log10(I[:, :, int(dmap.shape[2] / 2)] + 10)
        plt.imshow(dispim, interpolation='nearest', cmap='gray')
        fig.add_subplot(2, 3, 6)
        dispim = np.sum(rho, axis=2)
        plt.imshow(dispim, interpolation='nearest', cmap='gray')
    plt.show(block=False)
    # pg.mkQApp().exec_()


elif sel == 1:
    print('doing sel 1')

    s = 2  # Oversampling
    dtheta = wavelength / s / protein_diameter  # Angular step in theta

    d_vec = np.arange(1,10,1)*1e-10
    print(d_vec)

    num_d = len(d_vec)

    plt.figure()


    for ii in range(num_d):
        
        d = d_vec[ii]  # Autocorrelation ring resolution

        q = 2 * np.pi / d  # q magnitude of autocorrelation ring (2pi/d)
        n_phi = int(2 * np.pi * s * protein_diameter / d)  # Num angular bins in ring
        n_phi += (n_phi % 2)
        print('Nphi (num angular bins):', n_phi)

        dphi = 2 * np.pi / n_phi  # Angular step in phi
        phi = np.arange(n_phi) * dphi
        theta = 2 * np.arcsin(q * wavelength / 4 / np.pi)
        print(f'2 theta (deg): {2 * theta * 180 / np.pi:.3f}')

        st = np.sin(theta)
        sa = st * dphi * dtheta  # Ring bin solid angle
        q_ring = 2 * np.pi / wavelength * np.vstack(
            [st * np.cos(phi), st * np.sin(phi), (1 - np.cos(theta)) * np.ones(n_phi)]).T.copy()
        q_ring_gpu = clcore.to_device(q_ring)
        a_ring = clcore.to_device(shape=(n_phi,), dtype=clcore.complex_t) # This initialises zeros on the GPU of shape=shape

        acf_sum_noisy = np.zeros(n_phi)
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

            I_ring_noisy = np.random.poisson(I_ring).astype(np.float64)

            I_ring -= np.mean(I_ring)
            acf_sum += np.real(ifft(np.abs(fft(I_ring))**2))

            I_ring_noisy -= np.mean(I_ring_noisy)
            acf_sum_noisy += np.real(ifft(np.abs(fft(I_ring_noisy))**2))


        # plt.figure()
        # plt.plot(acf_sum)
        # plt.show(block=False)

        m = int(n_phi / 1)
        plt.plot(phi[1:m] * 180 / np.pi, acf_sum[1:m], label=f'd = {d}')


    plt.grid()
    plt.legend()
    plt.show(block=False)
    yay2




    m = int(n_phi / 2)

    plt.figure()
    plt.plot(phi[1:m] * 180 / np.pi, acf_sum[1:m], '-k')

    acf_sum_noisy += np.mean(acf_sum[1:m]) - np.mean(acf_sum_noisy[1:m])
    plt.plot(phi[1:m] * 180 / np.pi, acf_sum_noisy[1:m], '.r')

    plt.xlabel(r'$\Delta \phi$ (degrees)')
    plt.ylabel(r'$C(q, q, \Delta\phi)$')
    plt.show(block=False)
    # view_pad_data(pad_data=np.log10(I_sphere + 1), pad_geometry=pads, show=True)
    # view_pad_data(pad_data=np.random.poisson(np.log10(I_prot + 1)), pad_geometry=pads, show=True)

    # yay4