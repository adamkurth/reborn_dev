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

###################################################################
# Constants
##########################################################################
eV = const.eV
r_e = const.r_e
NA = const.N_A
h = const.h
c = const.c
water_density = 1000
rad90 = np.pi/2

##########################################################################
# Configurations
#######################################################################
pad_geometry_file = detector.epix100_geom_file
pad_binning = 4
photon_energy = 7000 * eV
detector_distance = 2.4
pulse_energy = 0.5e-3
drop_radius = 100e-9 / 2
beam_diameter = 0.5e-6
map_resolution = 0.2e-9  # Minimum resolution for 3D density map
map_oversample = 2  # Oversampling factor for 3D density map
cell = 200e-10  # Unit cell size (assume P1, cubic)
pdb_file = '1SS8' #'3IYF' '1PCQ' '2LYZ' 'BDNA25_sp.pdb'
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
n_shots = int(runtime * freq * hit_frac)
wavelength = h * c / photon_energy
beam = source.Beam(photon_energy=photon_energy, diameter_fwhm=beam_diameter, pulse_energy=pulse_energy)
beam.save_json('beam.json')
ddd
fluence = beam.photon_number_fluence
f_dens_water = atoms.xraylib_scattering_density('H2O', water_density, photon_energy, approximate=True)
pads = detector.load_pad_geometry_list(pad_geometry_file)
for p in pads:
    p.t_vec[2] = detector_distance
pads = pads.binned(pad_binning)
q_mags = pads.q_mags(beam=beam)
solid_angles = pads.solid_angles()
polarization_factors = pads.polarization_factors(beam=beam)
cryst = crystal.CrystalStructure(pdb_file, spacegroup='P1', unitcell=(cell, cell, cell, rad90, rad90, rad90))
dmap = crystal.CrystalDensityMap(cryst, map_resolution, map_oversample)
f = cryst.molecule.get_scattering_factors(beam=beam)
x = cryst.unitcell.r2x(cryst.molecule.get_centered_coordinates())
rho = dmap.place_atoms_in_map(x, f, mode='nearest')  # FIXME: replace 'nearest' with Cromer Mann densities
pg.image(np.sum(np.fft.fftshift(np.real(rho)), axis=2))
rho[rho != 0] -= f_dens_water * dmap.voxel_volume  # FIXME: Need a better model for solvent envelope
F = fftshift(fftn(rho))
I = np.abs(F) ** 2
rho_cell = fftshift(rho)
clcore = ClCore(double_precision=gpu_double_precision, group_size=gpu_group_size)
F_gpu = clcore.to_device(F)
q_vecs_gpu = clcore.to_device(pads.q_vecs(beam=beam))
q_mags_gpu = clcore.to_device(pads.q_mags(beam=beam))
amps_gpu = clcore.to_device(shape=pads.n_pixels, dtype=clcore.complex_t)
q_min = dmap.q_min
q_max = dmap.q_max
protein_number_density = protein_concentration/cryst.molecule.get_molecular_weight()
n_proteins_per_drop = int(protein_number_density*4/3*np.pi*drop_radius**3)
protein_diameter = cryst.molecule.max_atomic_pair_distance  # Nominal particle size
print('N Shots:', n_shots)
print('Molecules per drop:', n_proteins_per_drop)
print('Particle diameter:', protein_diameter)
print('Density map grid size: (%d, %d, %d)' % tuple(dmap.shape))

###########################################################################
# Water droplet with protein, 2D PAD simulation
##########################################################################
class DropletGetter(FrameGetter):
    def __init__(self):
        super().__init__()
        self.n_frames = np.inf
    def get_data(self, frame_number=0):
        dd = drop_radius*2 #+ (np.random.rand()-0.5)*drop_radius/5
        nppd = int(protein_number_density*4/3*np.pi*(dd/2)**3)
        p_vecs = placer.particles_in_a_sphere(sphere_diameter=dd, n_particles=nppd, particle_diameter=protein_diameter)
        add = False
        for p in range(nppd):
            R = Rotation.random().as_matrix()
            U = p_vecs[p, :]
            clcore.mesh_interpolation(F_gpu,q_vecs_gpu,N=dmap.shape,q_min=q_min,q_max=q_max,R=R,U=U,a=amps_gpu,add=add)
            add = True
        # clcore.sphere_form_factor(r=dd/2, q=q_mags_gpu, a=amps_gpu, add=add)
        sff = f_dens_water*sphere_form_factor(radius=dd/2, q_mags=q_mags)
        I = np.abs(amps_gpu.get()+sff)**2*r_e**2*solid_angles*polarization_factors*fluence
        I = np.abs(amps_gpu.get()) ** 2 * r_e ** 2 * solid_angles * polarization_factors * fluence
        I = np.random.poisson(I)
        df = dataframe.DataFrame(pad_geometry=pads, beam=beam, raw_data=I)
        return df
fg = DropletGetter()
pv = PADView(frame_getter=fg)
pv.start()
# sys.exit()

################################################################################
# FXS Simulation
################################################################################
s = 2  # Oversampling
d = 8e-10  # Autocorrelation ring resolution
q = 2 * np.pi / d  # q magnitude of autocorrelation ring
n_phi = int(2 * np.pi * s * protein_diameter / d)  # Num. bins in ring
n_phi += (n_phi % 2)
print('Nphi:', n_phi)
dphi = 2 * np.pi / n_phi  # Angular step in phi
phi = np.arange(n_phi) * dphi
dtheta = wavelength / s / protein_diameter  # Angular step in theta
theta = 2 * np.arcsin(q * wavelength / 4 / np.pi)
print('2 theta:', 2 * theta * 180 / np.pi)
st = np.sin(theta)
sa = st * dphi * dtheta  # Ring bin solid angle
q_ring = 2 * np.pi / wavelength * np.vstack(
    [st * np.cos(phi), st * np.sin(phi), (1 - np.cos(theta)) * np.ones(n_phi)]).T.copy()
q_ring_gpu = clcore.to_device(q_ring)
a_ring = clcore.to_device(shape=(n_phi,), dtype=clcore.complex_t)
acf_sum_noisy = np.zeros(n_phi)
acf_sum = np.zeros(n_phi)
for i in range(n_shots):
    if ((i + 1) % 1000) == 0:
        print(i + 1)
    p_vecs = placer.particles_in_a_sphere(sphere_diameter=drop_radius * 2, n_particles=n_proteins_per_drop, particle_diameter=protein_diameter)
    for p in range(n_proteins_per_drop):
        add = 1  # Default: add amplitudes of protein diffraction
        if p == 0:  # For the first molecule, do not add amplitudes.  Overwrite the GPU memory buffer instead.
            add = 0
        R = Rotation.random().as_matrix()
        U = p_vecs[p, :]
        clcore.mesh_interpolation(F_gpu, q_ring_gpu, N=dmap.shape, q_min=q_min, q_max=q_max, R=R, U=U, a=a_ring,
                                  add=add)
    I_ring = np.abs(a_ring.get()) ** 2
    I_ring *= sa * r_e ** 2 * fluence
    I_ring_noisy = np.random.poisson(I_ring).astype(np.float64)
    I_ring -= np.mean(I_ring)
    acf_sum += np.real(np.fft.ifft(np.abs(np.fft.fft(I_ring)) ** 2))
    I_ring_noisy -= np.mean(I_ring_noisy)
    acf_sum_noisy += np.real(np.fft.ifft(np.abs(np.fft.fft(I_ring_noisy)) ** 2))

m = int(n_phi / 2)
plt.plot(phi[1:m] * 180 / np.pi, acf_sum[1:m], '-k')
acf_sum_noisy += np.mean(acf_sum[1:m]) - np.mean(acf_sum_noisy[1:m])
plt.plot(phi[1:m] * 180 / np.pi, acf_sum_noisy[1:m], '.r')
plt.xlabel(r'$\Delta \phi$ (degrees)')
plt.ylabel(r'$C(q, q, \Delta\phi)$')
view_pad_data(pad_data=np.log10(I_sphere + 1), pad_geometry=pads, show=True)
view_pad_data(pad_data=np.random.poisson(np.log10(I_prot + 1)), pad_geometry=pads, show=True)
if 1:
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
plt.show()
pg.mkQApp().exec_()
