from time import time
import numpy as np
from numpy.fft import fftn, ifftn, fftshift
from scipy.spatial.transform import Rotation
import pyqtgraph as pg
from reborn import utils, source, detector, dataframe, const
from reborn.target import crystal, atoms, placer
from reborn.fileio.getters import FrameGetter
from reborn.simulate import solutions, gas, clcore
from reborn.viewers.qtviews import view_pad_data

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
droplet = 1  # Include droplet diffraction
correct_sa = 0  # Correct for solid angles (helpful for viewing multiple detectors with different pixel sizes)
atomistic = 0  # Do the all-atom protein simulation
gas_background = 1  # Include gas background
bulk_water = 1  # Include bulk water background
view = 1  # View diffraction
pad_geometry_file = [detector.epix100_geom_file, detector.jungfrau4m_geom_file]  # Can be list of detectors
pad_binning = [2, 2]  # For speed, bin the pixels
detector_distance = [2.4, 0.5]  # Average distances of detectors
beamstop_size = 5e-3  # Mask beamstop region
photon_energy = 7000 * eV  # All units are SI
pulse_energy = 0.5e-3
drop_radius = 100e-9 / 2  # Average droplet radius
drop_radius_fwhm = 0  # FWHM of droplet radius distribution (tophat distribution at present)
beam_diameter = 0.5e-6  # FWHM of x-ray beam (tophat profile at present)
map_resolution = 0.1e-9  # Minimum resolution for 3D density map.  Beware of overloading GPU memory...
map_oversample = 2  # Oversampling factor for 3D density map.  Beware of overloading GPU memory...
cell = 200e-10  # Unit cell size (assume P1, cubic).  Fake unit cell affects the density map.
pdb_file = ['1SS8', '3IYF', '1PCQ', '2LYZ', 'BDNA25_sp.pdb'][0]
protein_concentration = 10  # Protein concentration in mg/ml = kg/m^3
# Parameters for gas background calculation.  Give the full path through which x-rays interact with the gas.
gas_params = {'path_length': [0, None], 'gas_type': 'he', 'pressure': 100e-6, 'n_simulation_steps': 5,
              'temperature': 293}
random_seed = 2022  # Seed for random number generator (choose None to make it random)
gpu_double_precision = False
gpu_group_size = 32


#########################################################################
# Derived parameters
#########################################################################
if random_seed is not None:
    np.random.seed(random_seed)  # Make random numbers that are reproducible
# Set up geometry and beam
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
print("Set up density map from PDB file")
f_dens_water = atoms.xraylib_scattering_density('H2O', water_density, photon_energy, approximate=True)
q_mags = geom.q_mags(beam=beam)
cryst = crystal.CrystalStructure(pdb_file, spacegroup='P1', unitcell=(cell, cell, cell, rad90, rad90, rad90),
                                 create_bio_assembly=True)
dmap = crystal.CrystalDensityMap(cryst, map_resolution, map_oversample)
f = cryst.molecule.get_scattering_factors(beam=beam)
r_vecs = cryst.molecule.get_centered_coordinates()
x = cryst.unitcell.r2x(r_vecs)
rho = dmap.place_atoms_in_map(x, f, mode='nearest')  # FIXME: replace 'nearest' with Cromer Mann densities
rho[rho != 0] -= f_dens_water * dmap.voxel_volume  # FIXME: Need a better model for solvent envelope
F = fftshift(fftn(rho))
rho_cell = fftshift(rho)
print("Set up GPU")
gpucore = clcore.ClCore(double_precision=gpu_double_precision, group_size=gpu_group_size)
F_gpu = gpucore.to_device(F)
q_vecs_gpu = gpucore.to_device(geom.q_vecs(beam=beam))
q_mags_gpu = gpucore.to_device(geom.q_mags(beam=beam))
amps_gpu = gpucore.to_device(shape=geom.n_pixels, dtype=gpucore.complex_t)
q_min = dmap.q_min
q_max = dmap.q_max
protein_number_density = protein_concentration/cryst.molecule.get_molecular_weight()
n_proteins_per_drop = int(protein_number_density*4/3*np.pi*drop_radius**3)
protein_diameter = cryst.molecule.get_size()  # Nominal particle size

# pg.image(np.sum(np.fft.fftshift(np.real(rho)), axis=2))

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
    print("Simulate bulk water background")
    waterprof = solutions.water_scattering_factor_squared(q=q_mags)
    waterprof *= solutions.water_number_density()
    waterprof *= f2p

########################################################################
# Gas background.  This is the same for all shots.
########################################################################
if gas_background:
    print("Simulate gas background")
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
        t = time()
        np.random.seed(int(frame_number))
        dd = drop_radius*2 + (np.random.rand()-0.5)*drop_radius_fwhm
        nppd = int(protein_number_density*4/3*np.pi*(dd/2)**3)
        if one_particle:
            nppd = 1
        print('Simulating', nppd, 'particles')
        p_vecs = placer.particles_in_a_sphere(sphere_diameter=dd, n_particles=nppd, particle_diameter=protein_diameter)
        a_gpu = amps_gpu * 0
        if protein:
            for p in range(nppd):
                R = Rotation.random().as_matrix()
                U = p_vecs[p, :]
                if atomistic:
                    gpucore.phase_factor_qrf(q_vecs_gpu, r_vecs, f, R=R, U=U, a=a_gpu, add=True)
                else:
                    gpucore.mesh_interpolation(F_gpu, q_vecs_gpu, N=dmap.shape, q_min=q_min, q_max=q_max,
                                               R=R, U=U, a=a_gpu, add=True)
        if droplet > 0:
            gpucore.sphere_form_factor(r=dd / 2, q=q_mags_gpu, a=a_gpu, dens=f_dens_water, add=True)
        pattern = f2p * np.abs(a_gpu.get()) ** 2
        if gas_background:
            pattern += gasbak
        if bulk_water:
            pattern += 4*np.pi*(dd/2)**3/3 * waterprof
        if poisson:
            pattern = np.random.poisson(pattern)
        if correct_sa:
            pattern *= 1e6 / geom.solid_angles()
        print(time() - t, 'seconds')
        df = dataframe.DataFrame(pad_geometry=geom, beam=beam, raw_data=pattern, mask=mask)
        return df
print("Creating frame getter")
fg = DropletGetter()
print("Estimate time needed per pattern")
t0 = time()
for i in range(10):
    print(i)
    dat = fg.get_next_frame()
    intensities = dat.get_raw_data_flat()
    # intensities = dat.get_raw_data_list()
print((time()-t0)/10, 'seconds per pattern')
if view:
    pv = fg.get_padview(hold_levels=True, debug_level=3)
    pv.start()
