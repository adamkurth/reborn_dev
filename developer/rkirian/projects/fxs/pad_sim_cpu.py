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
conf = dict(
poisson = 1,  # Include Poisson noise
protein = 1,  # Include protein contrast in diffraction amplitudes
one_particle = 1,  # Fix particle counts to one per drop
droplet = 0,  # Include droplet diffraction
sheet = 0,  # Include sheet background (assume drop diameter is sheet thickness)
# correct_sa = 0,  # Correct for solid angles (helpful for viewing multiple detectors with different pixel sizes)
# atomistic = 0  # Do the all-atom protein simulation
gas_background = 0,  # Include gas background
solvent_contrast = 0,  # Include solvent contrast (water)
bulk_water = 0,  # Include bulk water background
use_cached_files = 1,  # Cache files for speed
pad_geometry_file = ["epix100_geometry.json", "jungfrau4m_geometry.json"],  # Can be list of detectors
pad_binning = [2, 2],  # For speed, bin the pixels
detector_distance = [2.4, 0.5],  # Average distances of detectors
beamstop_size = 5e-3,  # Mask beamstop region
photon_energy = 7000 * eV,  # All units are SI
pulse_energy = 1e-3,
drop_radius = 100e-9 / 2,  # Average droplet radius
drop_radius_fwhm = 0,  # FWHM of droplet radius distribution (tophat distribution at present)
beam_diameter = 110e-9,  # FWHM of x-ray beam (tophat profile at present)
map_resolution = 2e-10,  # Minimum resolution for 3D density map.
map_oversample = 4,  # Oversampling factor for 3D density map.
pdb_file = 'LargeCluster2nm.pdb',  # '1jb0' '1SS8' '3IYF' '1PCQ' '2LYZ' 'BDNA25_sp.pdb'
particle_name = '2nm Soot',  # Naming for plots
protein_concentration = 20,  # Protein concentration in mg/ml = kg/m^3
# Parameters for gas background calculation.  Give the full path through which x-rays interact with the gas.
gas_params = {'path_length': [0, None], 'gas_type': 'he', 'pressure': 100e-6, 'n_simulation_steps': 5,
              'temperature': 293},
random_seed = None, #2022  # Seed for random number generator (choose None to make it random)
fxs_ring_resolution = 8e-10,  # Autocorrelation ring resolution
fxs_ring_oversampling = 2,
fxs_n_shots = 100000
)
view_density_projections = 1,  # View reborn and DENSS densities (projected sum)
view_framegetter = 1,  # View diffraction

print('='*50)
print('INPUT PARAMETERS')
for k in conf.keys():
    print(k, ':', conf[k])
print('='*50)


#######################################################################
# Fetch files
########################################################################
if not os.path.exists(conf['pdb_file']):
    conf['pdb_file'] = crystal.get_pdb_file(conf['pdb_file'], save_path='.', use_cache=False)
    print('Fetched PDB file:', conf['pdb_file'])

#########################################################################
# Derived parameters
#########################################################################
if conf['random_seed'] is not None:
    np.random.seed(conf['random_seed'])  # Make random numbers that are reproducible
beam = source.Beam(photon_energy=conf['photon_energy'],
                   diameter_fwhm=conf['beam_diameter'],
                   pulse_energy=conf['pulse_energy'])
if not isinstance(conf['pad_geometry_file'], list):
    conf['pad_geometry_file'] = [conf['pad_geometry_file']]
    conf['detector_distance'] = [conf['detector_distance']]
    conf['pad_binning'] = [conf['pad_binning']]
geom = detector.PADGeometryList()
for i in range(len(conf['pad_geometry_file'])):
    p = detector.load_pad_geometry_list(conf['pad_geometry_file'][i]).binned(conf['pad_binning'][i])
    p.center_at_origin()
    p.translate([0, 0, conf['detector_distance'][i]])
    geom += p
mask = geom.beamstop_mask(beam=beam, min_radius=conf['beamstop_size'])
q_mags = geom.q_mags(beam=beam)
if conf['sheet']:
    conf['droplet'] = 0
liq_volume = 0

###############################################################
# Density map via reborn
##################################################################
cell = 100e-9
cryst = crystal.CrystalStructure(conf['pdb_file'], spacegroup='P1', unitcell=(cell, cell, cell, rad90, rad90, rad90),
                                 create_bio_assembly=True)
cell = cryst.molecule.max_atomic_pair_distance
cryst = crystal.CrystalStructure(conf['pdb_file'], spacegroup='P1', unitcell=(cell, cell, cell, rad90, rad90, rad90),
                                 create_bio_assembly=True)
dmap = crystal.CrystalDensityMap(cryst, conf['map_resolution'], conf['map_oversample'])
f = cryst.molecule.get_scattering_factors(beam=beam)
r_vecs = cryst.molecule.get_centered_coordinates()
x = cryst.unitcell.r2x(r_vecs)
rho = dmap.place_atoms_in_map(x, f, mode='nearest')  # FIXME: replace 'nearest' with Cromer Mann densities
f_dens_water = atoms.xraylib_scattering_density('H2O', water_density, conf['photon_energy'], approximate=True)
if conf['solvent_contrast']:
    f_dens_water = atoms.xraylib_scattering_density('H2O', water_density, conf['photon_energy'], approximate=True)
    rho[rho != 0] -= f_dens_water * dmap.voxel_volume  # FIXME: Need a better model for solvent envelope
# rho = ifftshift(rho)
if view_density_projections:
    pg.image(np.sum(rho.real, axis=2), title='reborn')

##################################################################
# Improved density map via DENSS
#################################################################
mrc_file = f"{conf['pdb_file']}_{conf['solvent_contrast']}_{conf['map_resolution']*1e10}_{conf['map_oversample']}.mrc"
if not os.path.exists(mrc_file):
    pdb = saxstats.PDBmod(conf['pdb_file'], bio_assembly=1)
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
    if conf['solvent_contrast']:
        rho = pdb2mrc.rho_insolvent
    else:
        rho = pdb2mrc.rho_invacuo
    # rho = ifftshift(rho)
    side = pdb2mrc.side
    if conf['use_cached_files']:
        print('Writing', mrc_file)
        saxstats.write_mrc(rho, side, mrc_file)
else:
    print('Loading', mrc_file)
    rho, side = saxstats.read_mrc(mrc_file)
rho = ifftshift(rho.real)
if view_density_projections:
    pg.image(np.sum(rho.real, axis=2), title='denss')
    pg.mkQApp().exec_()

##################################################################
# Prepare arrays for simulations
###################################################################
# rho = ifftshift(rho)
F = fftshift(fftn(rho)).copy()
rho_cell = fftshift(rho)
q_vecs = geom.q_vecs(beam=beam)
q_mags = geom.q_mags(beam=beam)
amps = np.zeros(geom.n_pixels, dtype=complex)
q_min = dmap.q_min
q_max = dmap.q_max
protein_number_density = conf['protein_concentration']/cryst.molecule.get_molecular_weight()
n_proteins_per_drop = int(protein_number_density*4/3*np.pi*conf['drop_radius']**3)
protein_diameter = cryst.molecule.max_atomic_pair_distance  # Nominal particle size

#####################################################################
# Conversion of structure factors |F(q)|^2 to photon counts.  This includes classical electron radius, incident fluence,
# solid angles of pixels, and polarization factor:  I(q) = r_e^2 J0 dOmega P(q) |F(q)|^2
##########################################################################
f2p = const.r_e**2 * beam.photon_number_fluence * geom.solid_angles() * geom.polarization_factors(beam=beam)

#######################################################################
# Bulk water profile.  Multiply this by the volume of water illuminated (i.e. droplet volume)
##########################################################################
if conf['bulk_water']:
    waterprof = solutions.water_scattering_factor_squared(q=q_mags)
    waterprof *= solutions.water_number_density()
    waterprof *= f2p

########################################################################
# Gas background.  This is the same for all shots.
########################################################################
if conf['gas_background']:
    gasbak = gas.get_gas_background(geom, beam, **conf['gas_params'])

######################################################################
# Sanity check: make proteins-in-drop amplitudes and inverse FT to see real-space image
########################################################################
if 0:
    # np.random.seed(0)
    dd = conf['drop_radius'] * 2
    vol = 4 / 3 * np.pi * (dd / 2) ** 3
    nppd = int(protein_number_density * vol)
    if conf['one_particle']:
        nppd = 1
    if conf['droplet']:
        p_vecs = placer.particles_in_a_sphere(sphere_diameter=dd, n_particles=nppd, particle_diameter=protein_diameter)
    else:
        p_vecs = placer.particles_in_a_cylinder(cylinder_diameter=dd, cylinder_length=dd, n_particles=nppd, particle_diameter=protein_diameter)
    q_max = 20 * 2*np.pi / (2*conf['drop_radius'])
    n = 112
    q = np.linspace(-q_max, q_max, n)
    qx, qy, qz = np.meshgrid(q, q, q, indexing='ij')
    qv = np.zeros((n**3, 3))
    qv[:, 0] = qx.ravel()
    qv[:, 1] = qy.ravel()
    qv[:, 2] = qz.ravel()
    del qx, qy, qz
    qm = reborn.utils.vec_mag(qv)
    amps = np.zeros(len(qm), dtype=complex)
    if conf['protein']:
        for p in range(nppd):
            R = Rotation.random().as_matrix()
            U = p_vecs[p, :]
            print('molecule', p, U)
            q = np.dot(qv, R)  # Rotate the protein (equivalent rotation on q, which is inverse of r rotation)
            q = qv
            a = trilinear_interpolation(F, q, x_min=q_min, x_max=q_max)
            #a *= np.exp(-1j * (np.dot(q, np.dot(U, R).T)))  # Shift the protein
            amps += a
    # amps += f_dens_water * sphere_form_factor(radius=dd / 2, q_mags=qm) / 1000
    amps = amps.reshape((n, n, n))
    print('FFT')
    rho2d = fftshift(ifftn(ifftshift(amps))).real
    print('Done')
    pg.image(np.sum(rho2d, axis=2))
    pg.mkQApp().exec_()
    # sys.exit()

###########################################################################
# Water droplet with protein, 2D PAD simulation.
# We make a subclass of the reborn "FrameGetter" base class to serve up DataFrames.
# This FrameGetter subclass can be replaced with the LCLSFrameGetter subclass for the analysis of real data.
##########################################################################
class PADGetter(FrameGetter):
    def __init__(self):
        super().__init__()
        self.n_frames = int(1e6)
    def get_data(self, frame_number=0):
        np.random.seed(int(frame_number))
        t = time()
        dd = conf['drop_radius'] * 2 + (np.random.rand() - 0.5) * conf['drop_radius_fwhm']
        if conf['droplet']:
            vol = 4 / 3 * np.pi * (dd / 2) ** 3
        else:
            vol = np.pi * (dd / 2) ** 2 * dd
        nppd = int(protein_number_density * vol)
        if conf['one_particle']:
            nppd = 1
        # print(nppd, 'particles in the drop')
        if conf['droplet']:
            p_vecs = placer.particles_in_a_sphere(sphere_diameter=dd, n_particles=nppd, particle_diameter=protein_diameter)
        else:
            p_vecs = placer.particles_in_a_cylinder(cylinder_diameter=dd, cylinder_length=dd, n_particles=nppd, particle_diameter=protein_diameter)
        amps = np.zeros(q_vecs.shape[0], dtype=complex)
        if conf['protein']:
            for p in range(nppd):
                # F(q) = FT(  sum_n f_n(R r + U)  ) = FT(  sum_n f_n(R r + U)  )
                R = Rotation.random().as_matrix()
                U = p_vecs[p, :]
                q = np.dot(q_vecs, R)  # Rotate the protein (equivalent rotation on q, which is inverse of r rotation)
                a = trilinear_interpolation(F, q, x_min=q_min, x_max=q_max)
                a *= np.exp(-1j * (np.dot(q, U.T)))  # Shift the protein
                amps += a
        print('Simulated protein amplitudes in', time() - t, 'seconds')
        if conf['droplet']:
            amps += f_dens_water * sphere_form_factor(radius=dd / 2, q_mags=q_mags)
        intensities = f2p * np.abs(amps) ** 2
        if conf['gas_background']:
            intensities += gasbak
        if conf['bulk_water']:
            intensities += 4 * np.pi * (dd / 2) ** 3 / 3 * waterprof
        if conf['poisson']:
            intensities = np.random.poisson(intensities)
        # if correct_sa:
        #     intensities *= 1e6 / geom.solid_angles()
        print('Simulated intensities in', time() - t, 'seconds')
        df = dataframe.DataFrame(pad_geometry=geom, beam=beam, raw_data=intensities, mask=mask)
        return df
fg = PADGetter()
if view_framegetter:
    hl = False
    if conf['poisson']:
        hl = True
    pv = fg.get_padview(hold_levels=hl)
    pv.start()

################################################################################
# FXS Simulation
################################################################################
res = conf['fxs_ring_resolution']
lam = beam.wavelength
s = conf['fxs_ring_oversampling']
n_shots = conf['fxs_n_shots']
d = protein_diameter  # Size of the object
q = 2 * np.pi / res
dq = 2*np.pi/d/s
theta = 2 * np.arcsin(q * lam / 4 / np.pi)
dtheta = lam / s / protein_diameter / np.cos(theta/2) # Angular step in theta
qperp = q * np.cos(theta/2)  # Component of qvec perpendicular to beam
dphi = dq / qperp
n_phi = int(2*np.pi/dphi)  # Num. bins in ring
n_phi += (n_phi % 2)  # Make it even
dphi = 2*np.pi/n_phi  # Update dphi to integer phis
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
t = time()
for i in range(n_shots):
    doprint = False
    if ((i + 1) % 10**np.ceil(np.log10(i+1))) == 0:
        doprint = True
    if doprint:
        print(f"Shot {i+1} @ {time()-t} seconds.")
    dd = conf['drop_radius'] * 2 + (np.random.rand() - 0.5) * conf['drop_radius_fwhm']
    if conf['droplet']:
        vol = 4 / 3 * np.pi * (dd / 2) ** 3
    else:
        vol = np.pi * (dd / 2) ** 2 * dd
    nppd = int(protein_number_density * vol)
    if conf['one_particle']:
        nppd = 1
    if conf['droplet']:
        p_vecs = placer.particles_in_a_sphere(sphere_diameter=dd, n_particles=nppd, particle_diameter=protein_diameter)
    else:
        p_vecs = placer.particles_in_a_cylinder(cylinder_diameter=dd, cylinder_length=dd, n_particles=nppd,
                                                particle_diameter=protein_diameter)
    amps = np.zeros(q_vecs.shape[0], dtype=complex)
    if conf['protein']:
        for p in range(nppd):
            R = Rotation.random().as_matrix()
            U = p_vecs[p, :]
            q = np.dot(q_vecs, R)
            a = trilinear_interpolation(F, q, x_min=q_min, x_max=q_max)
            a *= np.exp(-1j * (np.dot(q, U.T)))
            amps += a
    if conf['droplet']:
        amps += f_dens_water * sphere_form_factor(radius=dd / 2, q_mags=q_mags)
    intensities = f2p * np.abs(amps) ** 2
    if conf['bulk_water']:
        intensities += waterprof * vol
    if conf['gas_background']:
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
a = 1
plt.plot(phi[a:m] * 180 / np.pi, acf_sum[a:m], '-k')
acf_sum_noisy += np.mean(acf_sum[a:m]) - np.mean(acf_sum_noisy[a:m])  # set the noisy mean to match noise-free mean
plt.plot(phi[a:m] * 180 / np.pi, acf_sum_noisy[a:m], '.r')
plt.title(f'{conf["particle_name"]}, {res*1e10} $\AA$, {n_shots} shots')
plt.xlabel(r'$\Delta \phi$ (degrees)')
plt.ylabel(r'$C(q, q, \Delta\phi)$')
plt.show(block=True)
print('Done')
