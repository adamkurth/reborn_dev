from time import time
import numpy as np
import pyqtgraph as pg
from bornagain import detector
from bornagain import source
from bornagain import utils
from bornagain.utils import rotate, vec_norm
from bornagain.simulate.clcore import ClCore
from bornagain.simulate.examples import psi_pdb_file, lysozyme_pdb_file
from bornagain.target import crystal, density
from bornagain.viewers.qtviews import Scatter3D, bright_colors, colors
from bornagain.external.pyqtgraph.extras import keep_open


# Load up the pdb file for PSI
cryst = crystal.CrystalStructure(psi_pdb_file)
spacegroup = cryst.spacegroup
unitcell = cryst.unitcell

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
width = 2
length = 3
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


beam = source.Beam(wavelength=3e-10)
pad = detector.PADGeometry(pixel_size=100e-6, distance=1.0, n_pixels=1000)
q_vecs = pad.q_vecs(beam=beam)
clcore = ClCore()
t = time()
amps_dev = clcore.to_device(np.zeros(pad.shape(), dtype=clcore.complex_t))
amps_mol_dev = clcore.to_device(np.zeros(pad.shape(), dtype=clcore.complex_t))
amps_lat_dev = clcore.to_device(np.zeros(pad.shape(), dtype=clcore.complex_t))
h_vecs_dev = clcore.to_device(unitcell.q2h(q_vecs))
au_x_vecs_dev = clcore.to_device(au_x_coords, dtype=clcore.real_t)
au_f_dev = clcore.to_device(shape=(cryst.molecule.n_atoms,), dtype=clcore.real_t)*0 + 1
resolution = 2*np.pi/np.max(pad.q_mags(beam=beam))
oversampling = 4
dens = density.CrystalDensityMap(cryst=cryst, resolution=resolution, oversampling=oversampling)
amps3d_dev = clcore.to_device(shape=dens.shape, dtype=clcore.complex_t)*0
print('mesh', dens.shape)
clcore.phase_factor_mesh(au_x_vecs_dev, au_f_dev, dens.shape, q_min=dens.x_limits[:, 0], q_max=dens.x_limits[:, 1],
                         a=amps3d_dev)
print('done')

clcore.buffer_mesh_lookup(amps3d_dev, dens.shape, q_min=dens.x_limits[:, 0], q_max=dens.x_limits[:, 1],
                          q, R=None, a=None)
t = time()
for i in range(spacegroup.n_molecules):
    print('Symmetry partner %d' % (i,))
    mol_x_vecs = spacegroup.apply_symmetry_operation(i, au_x_coords)
    clcore.phase_factor_qrf(h_vecs_dev, mol_x_vecs, a=amps_mol_dev, add=False)
    clcore.phase_factor_qrf(h_vecs_dev, lats[i].occupied_x_coordinates, a=amps_lat_dev, add=False)
    amps_dev += amps_mol_dev * amps_lat_dev
intensities = pad.reshape(np.abs(amps_dev.get())**2)
print('GPU simulation: %g seconds' % (time()-t,))

pg.image(intensities * pad.beamstop_mask(beam=beam, min_angle=0.0001))
keep_open()
