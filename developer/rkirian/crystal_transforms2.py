from time import time
import numpy as np
import pyqtgraph as pg
import bornagain as ba
from bornagain import detector
from bornagain import source
from bornagain.utils import rotate
from bornagain.simulate.clcore import ClCore
from bornagain.simulate.examples import psi_pdb_file, lysozyme_pdb_file
from bornagain.target import crystal, density
from bornagain.viewers.qtviews import Scatter3D, bright_colors, colors
from bornagain.external.pyqtgraph.extras import keep_open


# Load up the pdb file for PSI
cryst = crystal.CrystalStructure(lysozyme_pdb_file)
spacegroup = cryst.spacegroup
unitcell = cryst.unitcell
r_vecs = cryst.molecule.coordinates.copy()
print(unitcell)

# Setup beam and detector
beam = source.Beam(wavelength=3e-10)
pad = detector.PADGeometry(pixel_size=300e-6, distance=0.2, n_pixels=300)
q_vecs = pad.q_vecs(beam=beam)


# Atomic scattering factors
f = ba.simulate.atoms.get_scattering_factors(cryst.molecule.atomic_numbers, ba.units.hc / beam.wavelength)*0 + 1

# Redefine operators so that trimers are tightly packed
spacegroup.sym_translations[2] += np.array([1, 0, 0])
spacegroup.sym_translations[4] += np.array([1, 1, 0])
spacegroup.sym_translations[3] += np.array([1, 1, 0])
spacegroup.sym_translations[5] += np.array([0, 1, 0])

# Coordinates of asymmetric unit in crystal basis x
au_x_coords = rotate(unitcell.o_mat_inv, cryst.molecule.coordinates)
au_x_com = np.mean(au_x_coords, axis=0)  # approximate center of mass

# Molecule centers of mass
mol_x_coms = list()
mean_com = 0
for i in range(spacegroup.n_molecules):
    com = spacegroup.apply_symmetry_operation(i, au_x_com)
    mean_com += com
    mol_x_coms.append(com)
mean_com /= spacegroup.n_molecules
for i in range(spacegroup.n_molecules):
    mol_x_coms[i] -= mean_com

# Construct the finite lattice generators
max_size = 41  # Make sure we have an odd value... for making hexagonal prisms
lats = [crystal.FiniteLattice(max_size=max_size, unitcell=unitcell) for i in range(spacegroup.n_molecules)]

# Construct a finite lattice in the form of a hexagonal prism
width = 1
length = 1
for i in range(spacegroup.n_molecules):
    lat = lats[i]
    com = mol_x_coms[i]
    lat.add_facet(plane=[-1, 1, 0], length=width, shift=com)
    lat.add_facet(plane=[1, -1, 0], length=width, shift=com)
    lat.add_facet(plane=[1, 0, 0], length=width, shift=com)
    lat.add_facet(plane=[0, 1, 0], length=width, shift=com)
    lat.add_facet(plane=[-1, 0, 0], length=width, shift=com)
    lat.add_facet(plane=[0, -1, 0], length=width, shift=com)
    lat.add_facet(plane=[0, 0, 1], length=length, shift=com)
    lat.add_facet(plane=[0, 0, -1], length=length, shift=com)

# Create a 3D scatterplot showing the crystal
viewcrystal = False
if viewcrystal:
    scat = Scatter3D()
    for i in range(spacegroup.n_molecules):
        x_vecs = lats[i].occupied_x_coordinates + mol_x_coms[i]
        scat.add_points(rotate(unitcell.o_mat, x_vecs), color=bright_colors(i, alpha=0.5), size=5)
        sym_vecs = spacegroup.apply_symmetry_operation(i, au_x_coords[0::1000, :])
        for j in range(x_vecs.shape[0]):
            lat_vec = lats[i].occupied_x_coordinates[j, :]
            r_vecs = rotate(unitcell.o_mat, sym_vecs - (mean_com + lat_vec))
            scat.add_points(r_vecs, color=bright_colors(i, alpha=0.1), size=1)
    scat.add_rgb_axis(length=100e-10)
    scat.show()

t = time()
h_vecs = unitcell.q2h(q_vecs)/2/np.pi
# print(np.max(h_vecs))

t = time()
clcore = ClCore()
# print(clcore.context.devices)
amps_gpu = clcore.to_device(shape=pad.shape(), dtype=clcore.complex_t)*0
amps_mol_gpu = clcore.to_device(shape=pad.shape(), dtype=clcore.complex_t)*0
amps_mol_gpu2 = clcore.to_device(shape=pad.shape(), dtype=clcore.complex_t)*0
amps_slice_gpu = clcore.to_device(shape=pad.shape(), dtype=clcore.complex_t)*0
amps_mol_interp_gpu = clcore.to_device(shape=pad.shape(), dtype=clcore.complex_t)*0
amps_lat_gpu = clcore.to_device(shape=pad.shape(), dtype=clcore.complex_t)*0
r_vecs_gpu = clcore.to_device(r_vecs, dtype=clcore.real_t)
q_vecs_gpu = clcore.to_device(q_vecs, dtype=clcore.real_t)
h_vecs_gpu = clcore.to_device(h_vecs, dtype=clcore.real_t)
au_x_vecs_gpu = clcore.to_device(au_x_coords, dtype=clcore.real_t)
au_f_gpu = clcore.to_device(f, dtype=clcore.complex_t)

resolution = 0.8*2*np.pi/np.max(pad.q_mags(beam=beam))
print('Resolution: %.3g A' % (resolution*1e10,))
oversampling = 2
dens = density.CrystalDensityMap(cryst=cryst, resolution=resolution, oversampling=oversampling)

qmax = 2 * np.pi / resolution
qmin = -qmax
N = np.ones(3, dtype=clcore.int_t)*100  # Number of samples
a_map_dev = clcore.to_device(shape=N, dtype=clcore.complex_t)
clcore.phase_factor_mesh(r_vecs, au_f_gpu, N=N, q_min=qmin, q_max=qmax, a=a_map_dev)

for i in range(spacegroup.n_molecules):
    print('Symmetry partner %d' % (i,))
    mol_x_vecs = spacegroup.apply_symmetry_operation(i, au_x_coords)
    clcore.phase_factor_qrf(2*np.pi*h_vecs_gpu, mol_x_vecs, au_f_gpu, a=amps_mol_gpu, add=True)
    clcore.phase_factor_qrf(q_vecs_gpu, unitcell.x2r(mol_x_vecs), au_f_gpu, a=amps_mol_gpu2, add=True)
    # clcore.phase_factor_qrf(2*np.pi*h_vecs_gpu, lats[i].occupied_x_coordinates, a=amps_lat_gpu, add=False)
    # RR = spacegroup.sym_rotations[i]
    # TT = spacegroup.sym_translations[i]
    clcore.buffer_mesh_lookup(a_map_dev, q_vecs_gpu, N=N, q_min=qmin, q_max=qmax, a=amps_slice_gpu, add=True)
    # amps_gpu += amps_mol_interp_gpu #* amps_lat_gpu

intensities1 = pad.reshape(np.abs(amps_slice_gpu.get()) ** 2)
intensities2 = pad.reshape(np.abs(amps_mol_gpu.get())**2)
intensities3 = pad.reshape(np.abs(amps_mol_gpu2.get())**2)
intensities = np.concatenate([intensities1, intensities2, intensities3])

print('GPU simulation: %g seconds' % (time()-t,))
pg.image(intensities)

keep_open()
