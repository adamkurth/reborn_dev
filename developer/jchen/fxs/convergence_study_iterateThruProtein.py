"""
Script to simulate fxs reconstruction

todo
from jugnfrau pull out smallest q - that will give the full q range

Date Created: 6 Jul 2022
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
# pdb_file = '2W0O' #'1SS8' #'4BED'#'1SS8'  # '3IYF' '1PCQ' '2LYZ' 'BDNA25_sp.pdb'
create_bio_assembly = 1

# protein_concentration = 10 #10  # Protein concentration in mg/ml = kg/m^3
# hit_frac = 0.01  # Hit fraction
# freq = 120  # XFEL frequency
# runtime = 0.1 * 3600  # Run time in seconds
random_seed = 2022  # Seed for random number generator (choose None to make it random)
gpu_double_precision = True
gpu_group_size = 32


"""
From Brent: As far as which samples are easiest to produce and are the most stable 
the order would be apoferritin >>> Beta galactosidase > GroEL.
"""
samples_list = ['2W0O', '6DRV', '1SS8']

protein_concentration = 10  # Protein concentration in mg/ml = kg/m^3

d = 8e-10 #8e-10  # Autocorrelation ring resolution

n_shots_ref = 1000 #10000
n_shots_tot = 1000 #10000
n_shot_mod = 100 # Calculate error every this many shots


savefile_name = 'autocorr_samples_1'


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

def autocorr_do(n_shots, n_proteins_per_drop, n_phi):

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



#--------------------------
# Convergence calculations

N_samples = len(samples_list)

errors = np.zeros((int(n_shots_tot / n_shot_mod), N_samples))


for i_samples in range(N_samples):
    pdb_file = samples_list[i_samples]

    #-------------------
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

    q_min = dmap.q_min
    q_max = dmap.q_max


    protein_diameter = cryst.molecule.max_atomic_pair_distance  # Nominal particle size

    # print('N Shots:', n_shots)
    # print('Molecules per drop:', n_proteins_per_drop)
    print('Particle diameter:', protein_diameter)
    print('Density map grid size: (%d, %d, %d)' % tuple(dmap.shape))
    print('f_dens_water: ', f_dens_water)


    protein_number_density = protein_concentration/cryst.molecule.get_molecular_weight()
    n_proteins_per_drop = int(np.round(protein_number_density*4/3*np.pi*drop_radius**3))


    ################################################################################
    # FXS Convergence study
    ################################################################################
    print('Begin convergence study')

    s = 2  # Oversampling

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



    print('calculating reference')
    acf_avg_ref = autocorr_do(n_shots=n_shots_ref, n_proteins_per_drop=1, n_phi=n_phi)



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

            errors[ind, i_samples] = (np.sum((acf_avg_curr[10:-10] - acf_avg_ref[10:-10])**2))
            ind += 1

errors = np.sqrt(errors) / np.sqrt(np.sum(acf_avg_ref[10:-10]**2))


# plt.figure()
# for i_samples in range(N_samples):
#     plt.plot(np.arange(1,len(errors)+1)*100, (errors[:,i_samples]), '-', label=f'{samples_list[i_samples]}')
# plt.xlabel('Num shots')
# plt.ylabel('RMS error to reference')
# # plt.ylim([0,50])
# plt.grid()
# plt.legend()
# plt.title(f'Autocorrelation ring resolution = {d}')
# plt.show(block=False)


np.savez(file=savefile_name,
         n_shots_ref=n_shots_ref,
         n_shots_tot=n_shots_tot,
         protein_concentration=protein_concentration,
         n_shot_mod=n_shot_mod,
         errors=errors,
         ring_oversampling=s,
         ring_res=d,
         water_density=water_density,
         pad_binning = pad_binning,
         photon_energy = photon_energy,
         detector_distance = detector_distance,
         pulse_energy = pulse_energy,
         drop_radius = drop_radius,
         beam_diameter = beam_diameter,
         map_resolution = map_resolution,
         map_oversample = map_oversample,
         cell = cell,
         create_bio_assembly = create_bio_assembly,
         random_seed = random_seed,
         gpu_double_precision = gpu_double_precision,
         gpu_group_size = gpu_group_size,
         samples_list=samples_list)



"""
# Load code
import numpy as np
import matplotlib.pyplot as plt

dat = np.load('autocorr_samples_1.npz')

n_shots_ref = dat['n_shots_ref']
n_shots_tot = dat['n_shots_tot']
protein_concentration = dat['protein_concentration']
n_shot_mod = dat['n_shot_mod']
errors = dat['errors']
water_density = dat['water_density']
d = dat['ring_res']

N_samples = len(samples_list)

plt.figure()
for i_samples in range(N_samples):
    plt.plot(np.arange(1,len(errors)+1)*100, (errors[:,i_samples]), '-', label=f'{samples_list[i_samples]}')
plt.xlabel('Num shots')
plt.ylabel('RMS error to reference')
# plt.ylim([0,50])
plt.grid()
plt.legend()
plt.title(f'Autocorrelation ring resolution = {d}')
plt.show(block=False)
"""






