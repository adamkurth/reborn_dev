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
cryst = crystal.CrystalStructure(lysozyme_pdb_file)
spacegroup = cryst.spacegroup
unitcell = cryst.unitcell
print(unitcell)

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
beam = source.Beam(wavelength=3e-10)
pad = detector.PADGeometry(pixel_size=200e-6, distance=0.2, n_pixels=500)
q_vecs = pad.q_vecs(beam=beam)
h_vecs = unitcell.q2h(q_vecs)/2/np.pi
print(np.max(h_vecs))

clcore = ClCore()
print(clcore.context.devices)
t = time()
amps_dev = clcore.to_device(np.zeros(pad.shape(), dtype=clcore.complex_t))
amps_mol_dev = clcore.to_device(np.zeros(pad.shape(), dtype=clcore.complex_t))
amps_mol_interp_dev = clcore.to_device(np.zeros(pad.shape(), dtype=clcore.complex_t))
amps_lat_dev = clcore.to_device(np.zeros(pad.shape(), dtype=clcore.complex_t))
h_vecs_dev = clcore.to_device(h_vecs, dtype=clcore.real_t)
au_x_vecs_dev = clcore.to_device(au_x_coords, dtype=clcore.real_t)
au_f_dev = clcore.to_device(shape=(cryst.molecule.n_atoms,), dtype=clcore.real_t)*0 + 1


resolution = 0.8*2*np.pi/np.max(pad.q_mags(beam=beam))
oversampling = 1
dens = density.CrystalDensityMap(cryst=cryst, resolution=resolution, oversampling=oversampling)
print('shape', dens.shape)
print('h dens', dens.h_vecs)
dens_h = dens.h_density_map
print('Resolution: %.3g A' % (resolution*1e10,))
print('hlims')
print(dens_h.limits)
print('H_vecs')
print(h_vecs)
amps3d_dev = clcore.to_device(shape=dens.shape, dtype=clcore.complex_t)*0
clcore.phase_factor_mesh(au_x_vecs_dev, au_f_dev, density_map=dens_h, a=amps3d_dev)
for i in range(1):#spacegroup.n_molecules):
    print('Symmetry partner %d' % (i,))
    mol_x_vecs = spacegroup.apply_symmetry_operation(i, au_x_coords)
    clcore.phase_factor_qrf(h_vecs_dev, mol_x_vecs, a=amps_mol_dev, add=False)
    clcore.phase_factor_qrf(h_vecs_dev, lats[i].occupied_x_coordinates, a=amps_lat_dev, add=False)
    RR = spacegroup.sym_rotations[i]
    TT = spacegroup.sym_translations[i]
    clcore.buffer_mesh_lookup(amps3d_dev, h_vecs_dev, density_map=dens_h, R=None, U=None, a=amps_mol_interp_dev, add=False)
    # amps_dev += amps_mol_interp_dev #* amps_lat_dev

intensities1 = pad.reshape(np.abs(amps_mol_interp_dev.get())**2)
intensities = intensities1
# intensities2 = pad.reshape(np.abs(amps_mol_dev.get())**2)
# intensities = np.concatenate([intensities1, intensities2])
print('GPU simulation: %g seconds' % (time()-t,))
pg.image(intensities)

keep_open()
