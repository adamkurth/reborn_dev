import os
import sys
from time import time
import numpy as np
from numpy.fft import fftn, ifftn, fftshift, ifftshift
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import pyqtgraph as pg
import reborn.utils
from reborn import source, detector, dataframe, const
from reborn.misc.interpolate import trilinear_interpolation
from reborn.simulate.form_factors import sphere_form_factor
from reborn.target import crystal, atoms, placer
from reborn.fileio.getters import FrameGetter
from reborn.simulate import solutions, gas, clcore
import saxstats.saxstats as saxstats  # This is Tom's DENSS package

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
poisson = 1  # Include Poisson noise
protein = 1  # Include protein contrast in diffraction amplitudes
one_particle = 0  # Fix particle counts to one per drop
droplet = 0  # Include droplet diffraction
sheet = 1  # Include sheet background (assume drop diameter is sheet thickness)
correct_sa = 0  # Correct for solid angles (helpful for viewing multiple detectors with different pixel sizes)
# atomistic = 0  # Do the all-atom protein simulation
gas_background = 0  # Include gas background
solvent_contrast = 1  # Include solvent contrast (water)
bulk_water = 0  # Include bulk water background
use_cached_files = 1  # Cache files for speed
view_framegetter = 1  # View diffraction
view_density_projections = 0  # View reborn and DENSS densities (projected sum)
pad_geometry_file = [detector.epix100_geom_file, detector.jungfrau4m_geom_file]  # Can be list of detectors
pad_binning = [2, 2]  # For speed, bin the pixels
detector_distance = [2.4, 0.5]  # Average distances of detectors
beamstop_size = 5e-3  # Mask beamstop region
photon_energy = 7000 * eV  # All units are SI
pulse_energy = 1e-3
drop_radius = 100e-9 / 2  # Average droplet radius
drop_radius_fwhm = 0  # FWHM of droplet radius distribution (tophat distribution at present)
beam_diameter = 110e-9  # FWHM of x-ray beam (tophat profile at present)
map_resolution = 8e-10  # Minimum resolution for 3D density map.  Beware of overloading GPU memory...
map_oversample = 4  # Oversampling factor for 3D density map.  Beware of overloading GPU memory...
cell = 200e-10  # Unit cell size (assume P1, cubic).  Fake unit cell affects the density map.
pdb_file = '1jb0'  #'1SS8', '3IYF', '1PCQ', '2LYZ', 'BDNA25_sp.pdb'
protein_concentration = 10  # Protein concentration in mg/ml = kg/m^3
# Parameters for gas background calculation.  Give the full path through which x-rays interact with the gas.
gas_params = {'path_length': [0, None], 'gas_type': 'he', 'pressure': 100e-6, 'n_simulation_steps': 5,
              'temperature': 293}
random_seed = 2022  # Seed for random number generator (choose None to make it random)

#######################################################################
# Fetch files
########################################################################
if not os.path.exists(pdb_file):
    pdb_file = crystal.get_pdb_file(pdb_file, save_path='.', use_cache=False)
    print('Fetched PDB file:', pdb_file)

#########################################################################
# Derived parameters
#########################################################################
if random_seed is not None:
    np.random.seed(random_seed)  # Make random numbers that are reproducible
beam = source.Beam(photon_energy=photon_energy, diameter_fwhm=beam_diameter, pulse_energy=pulse_energy)
if not isinstance(pad_geometry_file, list):
    pad_geometry_file = [pad_geometry_file]
    detector_distance = [detector_distance]
    pad_binning = [pad_binning]
geom = detector.PADGeometryList()
for i in range(len(pad_geometry_file)):
    p = detector.load_pad_geometry_list(pad_geometry_file[i]).binned(pad_binning[i])
    p.center_at_origin()
    p.translate([0, 0, detector_distance[i]])
    geom += p
mask = geom.beamstop_mask(beam=beam, min_radius=beamstop_size)
q_mags = geom.q_mags(beam=beam)
if sheet:
    droplet = 0
liq_volume = 0

###############################################################
# Density map via reborn
##################################################################
cryst = crystal.CrystalStructure(pdb_file, spacegroup='P1', unitcell=(cell, cell, cell, rad90, rad90, rad90),
                                 create_bio_assembly=True)
dmap = crystal.CrystalDensityMap(cryst, map_resolution, map_oversample)
f = cryst.molecule.get_scattering_factors(beam=beam)
r_vecs = cryst.molecule.get_centered_coordinates()
x = cryst.unitcell.r2x(r_vecs)
rho = dmap.place_atoms_in_map(x, f, mode='nearest')  # FIXME: replace 'nearest' with Cromer Mann densities
if solvent_contrast:
    f_dens_water = atoms.xraylib_scattering_density('H2O', water_density, photon_energy, approximate=True)
    rho[rho != 0] -= f_dens_water * dmap.voxel_volume  # FIXME: Need a better model for solvent envelope
if view_density_projections:
    pg.image(np.sum(rho.real, axis=2), title='reborn')

##################################################################
# Improved density map via DENSS
#################################################################
mrc_file = f"{pdb_file}_{solvent_contrast}_{map_resolution*1e10}_{map_oversample}.mrc"
if not os.path.exists(mrc_file):
    pdb = saxstats.PDBmod(pdb_file, bio_assembly=1)
    voxel = cell / dmap.cshape[0]
    side = dmap.shape[0] * voxel
    pdb2mrc = saxstats.PDB2MRC(pdb=pdb, voxel=voxel*1e10, side=side*1e10)
    pdb2mrc.scale_radii()
    pdb2mrc.make_grids()
    pdb2mrc.calculate_resolution()
    pdb2mrc.calculate_invacuo_density()
    pdb2mrc.calculate_excluded_volume()
    pdb2mrc.calculate_hydration_shell()
    pdb2mrc.calc_rho_with_modified_params(pdb2mrc.params)
    if solvent_contrast:
        rho = pdb2mrc.rho_insolvent
    else:
        rho = pdb2mrc.rho_invacuo
    rho = ifftshift(rho)
    side = pdb2mrc.side
    if use_cached_files:
        print('Writing', mrc_file)
        saxstats.write_mrc(rho, side, mrc_file)
else:
    print('Loading', mrc_file)
    rho, side = saxstats.read_mrc(mrc_file)
if view_density_projections:
    pg.image(np.sum(rho.real, axis=2), title='denss')
    pg.mkQApp().exec_()

##################################################################
# Prepare arrays for simulations
###################################################################
F = fftshift(fftn(rho)).copy()
rho_cell = fftshift(rho)
q_vecs = geom.q_vecs(beam=beam)
q_mags = geom.q_mags(beam=beam)
amps = np.zeros(geom.n_pixels, dtype=complex)
q_min = dmap.q_min
q_max = dmap.q_max
protein_number_density = protein_concentration/cryst.molecule.get_molecular_weight()
n_proteins_per_drop = int(protein_number_density*4/3*np.pi*drop_radius**3)
protein_diameter = cryst.molecule.max_atomic_pair_distance  # Nominal particle size

print('PDB:', pdb_file)
print('Molecules per drop:', n_proteins_per_drop)
print('Particle diameter:', protein_diameter)
print('Density map grid size: (%d, %d, %d)' % tuple(dmap.shape))

#####################################################################
# Conversion of structure factors |F(q)|^2 to photon counts.  This includes classical electron radius, incident fluence,
# solid angles of pixels, and polarization factor:  I(q) = r_e^2 J0 dOmega P(q) |F(q)|^2
##########################################################################
f2p = const.r_e**2 * beam.photon_number_fluence * geom.solid_angles() * geom.polarization_factors(beam=beam)

#######################################################################
# Bulk water profile.  Multiply this by the volume of water illuminated (i.e. droplet volume)
##########################################################################
if bulk_water:
    waterprof = solutions.water_scattering_factor_squared(q=q_mags)
    waterprof *= solutions.water_number_density()
    waterprof *= f2p

########################################################################
# Gas background.  This is the same for all shots.
########################################################################
if gas_background:
    gasbak = gas.get_gas_background(geom, beam, **gas_params)

###########################################################################
# Water droplet with protein, 2D PAD simulation.
# We make a subclass of the reborn "FrameGetter" base class to serve up DataFrames.
# This FrameGetter subclass can be replaced with the LCLSFrameGetter subclass for the analysis of real data.
##########################################################################
class DropletGetter(FrameGetter):
    def __init__(self):
        super().__init__()
        self.n_frames = int(1e6)
    def get_data(self, frame_number=0):
        np.random.seed(int(frame_number))
        t = time()
        dd = drop_radius * 2 + (np.random.rand() - 0.5) * drop_radius_fwhm
        if droplet:
            vol = 4 / 3 * np.pi * (dd / 2) ** 3
        else:
            vol = np.pi * (dd / 2) ** 2 * dd
        nppd = int(protein_number_density * vol)
        if one_particle:
            nppd = 1
        print(nppd, 'particles in the drop')
        if droplet:
            p_vecs = placer.particles_in_a_sphere(sphere_diameter=dd, n_particles=nppd, particle_diameter=protein_diameter)
        else:
            p_vecs = placer.particles_in_a_cylinder(cylinder_diameter=dd, cylinder_length=dd, n_particles=nppd, particle_diameter=protein_diameter)
        amps = np.zeros(q_vecs.shape[0], dtype=complex)
        if protein:
            for p in range(nppd):
                R = Rotation.random().as_matrix()
                U = p_vecs[p, :]
                q = np.dot(q_vecs, R)
                a = trilinear_interpolation(F, q, x_min=q_min, x_max=q_max)
                a *= np.exp(-1j * (np.dot(q, U.T)))
                amps += a
        print('Simulated protein amplitudes in', time() - t, 'seconds')
        if droplet:
            amps += f_dens_water * sphere_form_factor(radius=dd / 2, q_mags=q_mags)
        intensities = f2p * np.abs(amps) ** 2
        if gas_background:
            intensities += gasbak
        if bulk_water:
            intensities += 4 * np.pi * (dd / 2) ** 3 / 3 * waterprof
        if poisson:
            intensities = np.random.poisson(intensities)
        if correct_sa:
            intensities *= 1e6 / geom.solid_angles()
        print('Simulated intensities in', time() - t, 'seconds')
        df = dataframe.DataFrame(pad_geometry=geom, beam=beam, raw_data=intensities, mask=mask)
        return df
fg = DropletGetter()
if view_framegetter:
    hl = False
    if poisson:
        hl = True
    pv = fg.get_padview(hold_levels=hl)
    pv.start()

################################################################################
# FXS Simulation
################################################################################
lam = beam.wavelength
res = 20e-10  # Autocorrelation ring resolution
q = 2 * np.pi / res
s = 2  # Oversampling
d = protein_diameter  # Size of the object
dq = 2*np.pi/d/s
theta = 2 * np.arcsin(q * lam / 4 / np.pi)
dtheta = lam / s / protein_diameter / np.cos(theta/2) # Angular step in theta
qperp = q * np.cos(theta/2)  # Component of qvec perpendicular to beam
dphi = dq / qperp
n_phi = int(2*np.pi/dphi)  # Num. bins in ring
n_phi += (n_phi % 2)  # Make it even
dphi = 2*np.pi/n_phi  # Update dphi to integer phis
print('Nphi:', n_phi)
phi = np.arange(n_phi) * dphi
st = np.sin(theta)
sa = st * dphi * dtheta  # Ring bin solid angle
f2p = const.r_e**2 * beam.photon_number_fluence * sa  # Dropped polarization factor
q_vecs = 2 * np.pi / lam * np.vstack(
    [st * np.cos(phi), st * np.sin(phi), (1 - np.cos(theta)) * np.ones(n_phi)]).T.copy()
q_mags = reborn.utils.vec_mag(q_vecs)
a_ring = np.zeros(n_phi, dtype=complex)
acf_sum_noisy = np.zeros(n_phi)
acf_sum = np.zeros(n_phi)
waterprof = solutions.water_scattering_factor_squared(q=q)
waterprof *= f2p * solutions.water_number_density()
n_shots = 100000
t = time()
for i in range(n_shots):
    doprint = False
    if ((i + 1) % 10**np.ceil(np.log10(i+1))) == 0:
        doprint = True
    if doprint:
        print(f"Shot {i+1} @ {time()-t} seconds.")
    dd = drop_radius * 2 + (np.random.rand() - 0.5) * drop_radius_fwhm
    if droplet:
        vol = 4 / 3 * np.pi * (dd / 2) ** 3
    else:
        vol = np.pi * (dd / 2) ** 2 * dd
    nppd = int(protein_number_density * vol)
    if one_particle:
        nppd = 1
    if droplet:
        p_vecs = placer.particles_in_a_sphere(sphere_diameter=dd, n_particles=nppd, particle_diameter=protein_diameter)
    else:
        p_vecs = placer.particles_in_a_cylinder(cylinder_diameter=dd, cylinder_length=dd, n_particles=nppd,
                                                particle_diameter=protein_diameter)
    amps = np.zeros(q_vecs.shape[0], dtype=complex)
    if protein:
        for p in range(nppd):
            R = Rotation.random().as_matrix()
            U = p_vecs[p, :]
            q = np.dot(q_vecs, R)
            a = trilinear_interpolation(F, q, x_min=q_min, x_max=q_max)
            a *= np.exp(-1j * (np.dot(q, U.T)))
            amps += a
    if droplet:
        amps += f_dens_water * sphere_form_factor(radius=dd / 2, q_mags=q_mags)
    intensities = f2p * np.abs(amps) ** 2
    if bulk_water:
        intensities += waterprof * vol
    if gas_background:
        print('No gas background set up for FXS sims because of polarization factor handling')
        intensities += gasbak
    I_ring = intensities
    I_ring_noisy = np.random.poisson(I_ring).astype(np.float64)
    if doprint:
        print(f"{nppd} particles, {np.sum(I_ring)} photons in ring ({np.sum(I_ring)/n_phi} per pixel)")
    I_ring -= np.mean(I_ring)
    acf_sum += np.real(np.fft.ifft(np.abs(np.fft.fft(I_ring)) ** 2))
    I_ring_noisy -= np.mean(I_ring_noisy)
    acf_sum_noisy += np.real(np.fft.ifft(np.abs(np.fft.fft(I_ring_noisy)) ** 2))

m = int(n_phi / 2)
plt.plot(phi[1:m] * 180 / np.pi, acf_sum[1:m], '-k')
acf_sum_noisy += np.mean(acf_sum[1:m]) - np.mean(acf_sum_noisy[1:m])  # set the noisy mean to match noise-free mean
plt.plot(phi[1:m] * 180 / np.pi, acf_sum_noisy[1:m], '.r')
plt.title(f'PSI, {res*1e10} $\AA$, {n_shots} shots')
plt.xlabel(r'$\Delta \phi$ (degrees)')
plt.ylabel(r'$C(q, q, \Delta\phi)$')
plt.show(block=True)
print('Done')