import os
import sys
import time
import numpy as np
from numpy.fft import fftn, ifftn, fftshift
import pyqtgraph as pg
# import scipy.constants as const
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import saxstats.saxstats as saxs
print(saxs.__file__)
import reborn.target.crystal
from reborn import utils, source, detector, dataframe, const
from reborn.target import crystal, atoms, placer
from reborn.simulate.form_factors import sphere_form_factor
from reborn.fileio.getters import FrameGetter
from reborn.viewers.qtviews import view_pad_data, scatter_plot, PADView
from reborn.simulate.clcore import ClCore

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
# pad_geometry_file = detector.jungfrau4m_geom_file
pad_binning = 4
photon_energy = 7000 * eV
detector_distance = 0.58
pulse_energy = 1e-3
drop_radius = 100e-9 / 2
beam_diameter = 0.5e-6
map_resolution = 0.2e-9  # Minimum resolution for 3D density map
map_oversample = 2  # Oversampling factor for 3D density map
unit_cell_size = 200e-10  # Unit cell size (assume P1, cubic)
pdb_file = '1jb0.pdb'
solvent = True
# pdb_file = '1SS8'
# pdb_file = '3IYF' #'1PCQ' '2LYZ'
# pdb_file = 'BDNA25_sp.pdb'

protein_concentration = 10  # Protein concentration in mg/ml = kg/m^3
hit_frac = 0.01  # Hit fraction
freq = 120  # XFEL frequency
runtime = 0.1 * 3600  # Run time in seconds
random_seed = 2022  # Seed for random number generator (choose None to make it random)
gpu_double_precision = True
gpu_group_size = 32
poisson = False

######################################################################
# PDB Files
########################################################################
if not os.path.exists(pdb_file):
    out = reborn.target.crystal.get_pdb_file(pdb_file, save_path='.', use_cache=False)

#########################################################################
# Derived parameters
#########################################################################
if random_seed is not None:
    np.random.seed(random_seed)  # Make random numbers that are reproducible
n_shots = int(runtime * freq * hit_frac)
wavelength = h * c / photon_energy
beam = source.Beam(photon_energy=photon_energy, diameter_fwhm=beam_diameter, pulse_energy=pulse_energy)
beam.save_json('beam.json')
fluence = beam.photon_number_fluence
pads = detector.load_pad_geometry_list(pad_geometry_file)
pads.set_average_detector_distance(detector_distance, beam=beam)
pads = pads.binned(pad_binning)
q_mags = pads.q_mags(beam=beam)
solid_angles = pads.solid_angles()
polarization_factors = pads.polarization_factors(beam=beam)

######################################################
# reborn density map
#####################################################
# Even if we are using a DENSS map, we still need this for the configuration of samplings, etc.  Shouldn't be
# necessary; it's a crutch for now...
print('reborn map')
# We are working in the crystal basis here, even though there are no crystals involved.  But we have modified
# the unit cell to be orthogonal, and the spacegroup to have no symmetry, which emulates a typical cartesian grid.
cryst = crystal.CrystalStructure(pdb_file, create_bio_assembly=True, spacegroup='P1',
                                 unitcell=(unit_cell_size, unit_cell_size, unit_cell_size, rad90, rad90, rad90))
dmap = crystal.CrystalDensityMap(cryst, map_resolution, map_oversample)
voxel = unit_cell_size / dmap.cshape[0]
side = dmap.shape[0]*voxel
print('voxel', voxel)
print('side', side)
print('n', side/voxel)
f = cryst.molecule.get_scattering_factors(beam=beam)
x = cryst.unitcell.r2x(cryst.molecule.get_centered_coordinates())  # Atom coordinates in **crystal basis**
rho = dmap.place_atoms_in_map(x, f, mode='nearest')  # FIXME: replace 'nearest' with Cromer Mann densities
if solvent:
    f_dens_water = atoms.xraylib_scattering_density('H2O', water_density, photon_energy, approximate=True)
    rho[rho != 0] -= f_dens_water * dmap.voxel_volume  # FIXME: Need a better model for solvent envelope
rho = fftshift(rho)
print('sum rho', np.sum(rho))
pg.image(np.sum(rho.real, axis=2), title='reborn')
# pg.mkQApp().exec_()

######################################################
# DENSS density map
#######################################################
mrc_file = pdb_file + ".mrc"
if 1: #not os.path.exists(mrc_file):
    pdb = saxs.PDBmod(pdb_file, bio_assembly=1)
    # n = int(cryst.molecule.coordinates.shape[0] / pdb.coords.shape[0])
    # if n > 1:
    #     pdb.coords = cryst.molecule.coordinates*1e10
    #     pdb.atomnum = np.concatenate([pdb.atomnum]*n)
    #     pdb.atomname = np.concatenate([pdb.atomname]*n)
    #     pdb.atomalt = np.concatenate([pdb.atomalt]*n)
    #     pdb.resname = np.concatenate([pdb.resname]*n)
    #     pdb.resnum = np.concatenate([pdb.resnum]*n)
    #     pdb.chain = np.concatenate([pdb.chain]*n)
    #     pdb.occupancy = np.concatenate([pdb.occupancy]*n)
    #     pdb.b = np.concatenate([pdb.b]*n)
    #     pdb.atomtype = np.concatenate([pdb.atomtype]*n)
    #     pdb.charge = np.concatenate([pdb.charge]*n)
    #     pdb.nelectrons = np.concatenate([pdb.nelectrons]*n)
    #     # pdb.radius = np.concatenate([pdb.radius]*n)
    #     pdb.vdW = np.concatenate([pdb.vdW]*n)
    #     pdb.unique_volume = np.concatenate([pdb.unique_volume]*n)
    #     pdb.unique_radius = np.concatenate([pdb.unique_radius]*n)
    #     pdb.numH = np.concatenate([pdb.numH]*n)
    #     pdb.natoms = pdb.natoms * n
    #     print(pdb.coords.shape)
    # print(cryst.molecule.coordinates.shape)
    # print(pdb.coords[0, :])
    # print(cryst.molecule.coordinates[0, :]*1e10)
    # sys.exit()
    pdb2mrc = saxs.PDB2MRC(pdb=pdb, voxel=voxel*1e10, side=side*1e10)
    pdb2mrc.scale_radii()
    pdb2mrc.make_grids()
    pdb2mrc.calculate_resolution()
    pdb2mrc.calculate_invacuo_density()
    pdb2mrc.calculate_excluded_volume()
    pdb2mrc.calculate_hydration_shell()
    pdb2mrc.calc_rho_with_modified_params(pdb2mrc.params)
    print('voxel', pdb2mrc.dx)
    print('side', pdb2mrc.side)
    print('n', pdb2mrc.side / pdb2mrc.dx)
    if solvent:
        rho = pdb2mrc.rho_insolvent #/ pdb2mrc.dV
    else:
        rho = pdb2mrc.rho_invacuo #/ pdb2mrc.dV
    print('sum rho', np.sum(rho))
    side = pdb2mrc.side
    # saxs.write_mrc(rho, side, mrc_file)
else:
    rho, side = saxs.read_mrc(mrc_file)
print('denss map')
pg.image(np.sum(rho.real, axis=2), title='denss')
pg.mkQApp().exec_()

####################################################
# Prepare GPU etc.
######################################################
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
n_proteins_per_drop = 1 #int(protein_number_density*4/3*np.pi*drop_radius**3)
protein_diameter = cryst.molecule.max_atomic_pair_distance  # Nominal particle size
print('N Shots:', n_shots)
print('Molecules per drop:', n_proteins_per_drop)
print('Particle diameter:', protein_diameter)
print('Density map grid size: (%d, %d, %d)' % tuple(dmap.shape))

###########################################################################
# Water droplet with protein, 2D PAD simulation
##########################################################################
class Simulator(FrameGetter):
    def __init__(self):
        super().__init__()
        self.n_frames = 1e6
    def get_data(self, frame_number=0):
        dd = drop_radius*2 #+ (np.random.rand()-0.5)*drop_radius/5
        nppd = n_proteins_per_drop # int(protein_number_density*4/3*np.pi*(dd/2)**3)
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
        if poisson:
            I = np.random.poisson(I)
        df = dataframe.DataFrame(pad_geometry=pads, beam=beam, raw_data=I)
        return df
fg = Simulator()
pv = PADView(frame_getter=fg)
pv.start()
sys.exit()

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
