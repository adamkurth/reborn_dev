import sys
from time import time
import numpy as np
import pyqtgraph as pg
from bornagain import detector
from bornagain import source
from bornagain.utils import trilinear_insert, random_rotation, vec_mag, vec_norm
from bornagain.simulate.clcore import ClCore
from bornagain.simulate.examples import psi_pdb_file, lysozyme_pdb_file
from bornagain.target import crystal, density
from bornagain.viewers.qtviews import Scatter3D, bright_colors, colors, PADView
from bornagain.external.pyqtgraph import keep_open
import scipy.constants as const
import argparse

eV = const.value('electron volt')
r_e = const.value("classical electron radius")



parser = argparse.ArgumentParser('Simulate finite crystals and merge intensities into a 3D map.')
parser.add_argument('--n_pixels', type=int, default=256, required=False,
                    help='Make square patterns of this size')
parser.add_argument('--pixel_size', type=float, default=100.0e-6, required=False,
                    help='Size of the square pixels')
parser.add_argument('--detector_distance', type=float, default=1.0, required=False,
                    help='Distance to the detector')
parser.add_argument('--n_patterns', type=int, default=1, required=False, 
                    help='How many patterns to simulate')
parser.add_argument('--photon_energy_ev', type=float, default=8000, required=False,
                    help='Photon energy in electron volts')
parser.add_argument('--pulse_energy', type=float, default=1e-6, required=False,
                    help='X-ray beam diameter in meters')
parser.add_argument('--beam_diameter', type=float, default=1e-3, required=False,
                    help='X-ray pulse energy in Joules')
parser.add_argument('--checkpoint_save_interval', type=int, default=1e10, required=False, 
                    help='Save curent results at this interval')
parser.add_argument('--view_crystal', action='store_false', required=False,
                    help='Display the crystal molecule coordinates')  # 'store_false' means True by default
parser.add_argument('--add_facets', action='store_false', required=False,
                    help='Add facets to the crystals')   # This means True by default
parser.add_argument('--run_number', type=int, default=1, required=False,
                    help='Files will be prefixed with "run%%04d_"')
parser.add_argument('--pdb_file', type=str, default='1jb0', required=False,
                    help='PDB file path or ID')
parser.add_argument('--random_orientations', action='store_false', required=False,
                    help='Randomize crystal orientations')
args = parser.parse_args()


photon_energy = args.photon_energy_ev * eV

# Load up the pdb file
cryst = crystal.CrystalStructure(args.pdb_file, tight_packing=True)  # Tight packing: put all molecules in unit cell

# Coordinates of asymmetric unit in crystal basis x
au_x_vecs = cryst.fractional_coordinates
# Redefine spacegroup operators so that molecules COMs are tightly packed in the unit cell
au_x_com = cryst.fractional_coordinates_com
mol_x_coms = np.zeros((cryst.spacegroup.n_molecules, 3))
mol_r_coms = np.zeros((cryst.spacegroup.n_molecules, 3))
for i in range(cryst.spacegroup.n_molecules):
    com = cryst.spacegroup.apply_symmetry_operation(i, au_x_com)
    # cryst.spacegroup.sym_translations[i] -= com - (com % 1)
    mol_x_coms[i, :] = cryst.spacegroup.apply_symmetry_operation(i, au_x_com)
    mol_r_coms[i, :] = cryst.unitcell.x2r(cryst.spacegroup.apply_symmetry_operation(i, au_x_com))

# Setup beam and detector
beam = source.Beam(photon_energy=photon_energy, pulse_energy=args.pulse_energy, diameter_fwhm=args.beam_diameter)
pad = detector.PADGeometry(pixel_size=args.pixel_size, distance=args.detector_distance, shape=[args.n_pixels]*2)
q_vecs = pad.q_vecs(beam=beam)
q_max = np.max(pad.q_mags(beam=beam))
resolution = 2*np.pi/q_max
print('Resolution: %.3g A' % (resolution*1e10,))
mask = pad.beamstop_mask(beam=beam, q_min=2*np.pi/500e-10)
scale_to_photon_counts = pad.reshape(r_e**2 * pad.solid_angles() * pad.polarization_factors(beam=beam) *
                                     beam.photon_number_fluence)

# Density map for merging
# dmap = crystal.CrystalDensityMap(cryst=cryst, resolution=resolution, oversampling=10)
# dens = np.zeros(dmap.shape)
# denswt = np.zeros(dmap.shape)

# Atomic scattering factors
f = cryst.molecule.get_scattering_factors(beam=beam)

# Construct the finite lattice generators
max_size = 41  # Make sure we have an odd value... for making hexagonal prisms
lats = [crystal.FiniteLattice(max_size=max_size, unitcell=cryst.unitcell) for i in range(cryst.spacegroup.n_molecules)]

clcore = ClCore()
amps_gpu = clcore.to_device(shape=pad.shape(), dtype=clcore.complex_t)*0
amps_mol_gpu = clcore.to_device(shape=pad.shape(), dtype=clcore.complex_t)
amps_lat_gpu = clcore.to_device(shape=pad.shape(), dtype=clcore.complex_t)
q_vecs_gpu = clcore.to_device(q_vecs, dtype=clcore.real_t)
q_rot_gpu = clcore.to_device(shape=q_vecs.shape, dtype=clcore.real_t)
# g_vecs_gpu = clcore.to_device(q_vecs, dtype=clcore.real_t)
h_rot_gpu = clcore.to_device(shape=q_vecs.shape, dtype=clcore.real_t)
f_gpu = clcore.to_device(f, dtype=clcore.complex_t)

merge_sum = 0
weight_sum = 0

h_max = 30
h_min = -30
n_h_bins = (h_max-h_min)*4*np.ones([3])+1
h_corner_min = h_min*np.ones([3])
h_corner_max = h_max*np.ones([3])

# Put all the atomic coordinates on the gpu
mol_vecs = []
mol_r_vecs_gpu = []
for i in range(cryst.spacegroup.n_molecules):
    mv = cryst.spacegroup.apply_symmetry_operation(i, au_x_vecs)
    mv = cryst.unitcell.x2r(mv)
    mol_vecs.append(mv)
    mol_r_vecs_gpu.append(clcore.to_device(mv, dtype=clcore.real_t))

scat = None

for c in range(args.n_patterns):
    # Rotate the lattice basis vectors
    rot = random_rotation()
    rot = np.array([[-0.154, 0.350, 0.923], [0.700, 0.698, -0.148], [-0.697, 0.623, -0.353]])
    trans = np.zeros(3)
    A = np.dot(rot, cryst.unitcell.a_mat)
    # Construct a finite lattice in the form of a hexagonal prism
    width = 2 + np.random.rand(1)*3
    length = 5 + np.random.rand(1)*3
    if args.add_facets:
        for i in range(cryst.spacegroup.n_molecules):
            lat = lats[i]
            com = mol_x_coms[i, :]
            lat.add_facet(plane=[-1, 1, 0], length=width, shift=com)
            lat.add_facet(plane=[1, -1, 0], length=width, shift=com)
            lat.add_facet(plane=[1, 0, 0], length=width, shift=com)
            lat.add_facet(plane=[0, 1, 0], length=width, shift=com)
            lat.add_facet(plane=[-1, 0, 0], length=width, shift=com)
            lat.add_facet(plane=[0, -1, 0], length=width, shift=com)
            lat.add_facet(plane=[0, 0, 1], length=length, shift=com)
            lat.add_facet(plane=[0, 0, -1], length=length, shift=com)
    # For viewing the crystal lattice and molecule coordinates
    if args.view_crystal:
        if scat is not None:
            del scat
        scat = Scatter3D()
    t = time()
    clcore.rotate_translate_vectors(rot, trans, q_vecs_gpu, q_rot_gpu)
    clcore.rotate_translate_vectors(rot, trans, q_vecs_gpu, h_rot_gpu)
    amps_gpu *= 0
    for i in range(cryst.spacegroup.n_molecules):
        lat_vecs = lats[i].occupied_x_coordinates # + mol_x_coms[i, :]
        # lat_vecs = cryst.unitcell.x2r(lat_vecs)
        clcore.phase_factor_qrf(q_rot_gpu, lat_vecs, a=amps_lat_gpu, add=False)  # All lattice cites
        # mol_vecs = spacegroup.apply_symmetry_operation(i, au_x_vecs)
        # mol_vecs = cryst.unitcell.x2r(mol_vecs)
        clcore.phase_factor_qrf(q_rot_gpu, mol_r_vecs_gpu[i], f_gpu, a=amps_mol_gpu, add=False)  # All atoms in molecule...
        amps_gpu += amps_lat_gpu * amps_mol_gpu
        if args.view_crystal:
            scat.add_points(lat_vecs + mol_r_coms[i, :], color=bright_colors(i, alpha=0.5), size=5)
            scat.add_points(mol_vecs[i], color=bright_colors(i, alpha=0.5), size=1)
    if args.view_crystal:
        scat.add_rgb_axis()
        scat.add_unit_cell(cell=cryst.unitcell)
        scat.set_orthographic_projection()
        scat.show()
    intensities = scale_to_photon_counts * np.abs(amps_gpu.get()) ** 2
    # intensities = np.random.poisson(intensities).astype(np.float64)
    h_vecs = cryst.unitcell.q2h(q_rot_gpu.get())
    merge, weight = trilinear_insert(h_vecs, intensities.ravel(), h_corner_min, h_corner_max, n_h_bins, mask2.ravel())
    merge_sum += merge
    weight_sum += weight
    print('Pattern %d; %.3f seconds' % (c, time()-t,))
    if ((c+1) % args.checkpoint_save_interval) == 0:
        print('saving checkpoint on pattern %d run %d' % (c+1, args.run_number))
        np.savez('run%04d_checkpoint%06d.npz' % (args.run_number, c+1), merge_sum=merge_sum, weight_sum=weight_sum,
                 h_corner_min=h_corner_min, h_corner_max=h_corner_max, n_h_bins=n_h_bins)

if True:
    dat = intensities*mask
    padview = PADView(pad_geometry=[pad], raw_data=[dat])
    padview.show_coordinate_axes()
    padview.set_levels(0, np.percentile(dat, 90))
    padview.start()

# w = np.where(weight_sum > 0)[0]
# merge_avg = merge_sum.copy()
# merge_avg.flat[w] /= weight_sum.flat[w]
# pg.image(merge_avg)
