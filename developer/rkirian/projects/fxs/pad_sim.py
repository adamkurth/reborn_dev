from time import time
import numpy as np
from numpy.fft import fftn, ifftn, fftshift
from scipy.spatial.transform import Rotation
import pyqtgraph as pg
from reborn import utils, source, detector, dataframe, const
from reborn.target import crystal, atoms, placer
from reborn.fileio.getters import FrameGetter
from reborn.simulate import gas, clcore
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
pad_geometry_file = [detector.epix100_geom_file, detector.jungfrau4m_geom_file]
pad_binning = [4, 1]
detector_distance = [2.4, 0.5]
beamstop_size = 5e-3
photon_energy = 7000 * eV
pulse_energy = 0.5e-3
drop_radius = 70e-9 / 2
beam_diameter = 0.5e-6
map_resolution = 0.2e-9  # Minimum resolution for 3D density map
map_oversample = 2  # Oversampling factor for 3D density map
cell = 200e-10  # Unit cell size (assume P1, cubic)
pdb_file = ['1SS8', '3IYF', '1PCQ', '2LYZ', 'BDNA25_sp.pdb'][0]
protein_concentration = 10  # Protein concentration in mg/ml = kg/m^3
random_seed = 2022  # Seed for random number generator (choose None to make it random)
gpu_double_precision = True
gpu_group_size = 32
poisson = 0
protein = 1
droplet = 0
correct_sa = 1
atomistic = 0
one_particle = 1
gas_background = 0
view = 1

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
pads = detector.PADGeometryList()
for i in range(len(pad_geometry_file)):
    p = detector.load_pad_geometry_list(pad_geometry_file[i]).binned(pad_binning[i])
    p.center_at_origin()
    p.translate([0, 0, detector_distance[i]])
    pads += p
mask = pads.beamstop_mask(beam=beam, min_radius=beamstop_size)
f2phot = pads.f2phot(beam)
f_dens_water = atoms.xraylib_scattering_density('H2O', water_density, photon_energy, approximate=True)
q_mags = pads.q_mags(beam=beam)
cryst = crystal.CrystalStructure(pdb_file, spacegroup='P1', unitcell=(cell, cell, cell, rad90, rad90, rad90))
dmap = crystal.CrystalDensityMap(cryst, map_resolution, map_oversample)
f = cryst.molecule.get_scattering_factors(beam=beam)
r_vecs = cryst.molecule.get_centered_coordinates()
x = cryst.unitcell.r2x(r_vecs)
rho = dmap.place_atoms_in_map(x, f, mode='nearest')  # FIXME: replace 'nearest' with Cromer Mann densities
rho[rho != 0] -= f_dens_water * dmap.voxel_volume  # FIXME: Need a better model for solvent envelope
F = fftshift(fftn(rho))
I = np.abs(F) ** 2
rho_cell = fftshift(rho)
gpucore = clcore.ClCore(double_precision=gpu_double_precision, group_size=gpu_group_size)
F_gpu = gpucore.to_device(F)
q_vecs_gpu = gpucore.to_device(pads.q_vecs(beam=beam))
q_mags_gpu = gpucore.to_device(pads.q_mags(beam=beam))
amps_gpu = gpucore.to_device(shape=pads.n_pixels, dtype=gpucore.complex_t)
q_min = dmap.q_min
q_max = dmap.q_max
protein_number_density = protein_concentration/cryst.molecule.get_molecular_weight()
n_proteins_per_drop = int(protein_number_density*4/3*np.pi*drop_radius**3)
protein_diameter = cryst.molecule.max_atomic_pair_distance  # Nominal particle size

# pg.image(np.sum(np.fft.fftshift(np.real(rho)), axis=2))

print('PDB:', pdb_file)
print('Molecules per drop:', n_proteins_per_drop)
print('Particle diameter:', protein_diameter)
print('Density map grid size: (%d, %d, %d)' % tuple(dmap.shape))

if gas_background:
    gass = gas.get_gas_background(pads, beam, path_length=[-0, 2.5], gas_type='he', pressure=100e-5, n_simulation_steps=5)
    gass2 = gas.get_gas_background(pads, beam, path_length=[-0, 5], gas_type='he', pressure=100e-5, n_simulation_steps=10)
    print('test', np.max(np.abs((gass-gass2)/gass)))
    if view:
        view_pad_data(pad_geometry=pads, pad_data=gass, beam=beam)



###########################################################################
# Water droplet with protein, 2D PAD simulation
##########################################################################
class DropletGetter(FrameGetter):
    def __init__(self):
        super().__init__()
        self.n_frames = int(1e6)
    def get_data(self, frame_number=0):
        t = time()
        np.random.seed(int(frame_number))
        dd = drop_radius*2 #+ (np.random.rand()-0.5)*drop_radius/5
        nppd = int(protein_number_density*4/3*np.pi*(dd/2)**3)
        if one_particle:
            nppd = 1
        print(nppd, 'particles')
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
        I = 0
        if droplet > 0:
            gpucore.sphere_form_factor(r=dd / 2, q=q_mags_gpu, a=a_gpu, dens=f_dens_water, add=True)
        I = np.abs(a_gpu.get()) ** 2 * f2phot
        if gas_background:
            I += gass
        if poisson:
            I = np.random.poisson(I)
        if correct_sa:
            I *= 1e6/pads.solid_angles()
        print(time() - t, 'seconds')
        df = dataframe.DataFrame(pad_geometry=pads, beam=beam, raw_data=I, mask=mask)
        return df
fg = DropletGetter()
df = fg.get_next_frame()
if view:
    pv = fg.get_padview(hold_levels=True)
    pv.start()