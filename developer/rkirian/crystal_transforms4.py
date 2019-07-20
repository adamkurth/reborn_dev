from time import time
import numpy as np
import pyqtgraph as pg
from bornagain import detector
from bornagain import source
from bornagain.utils import random_rotation, vec_mag, vec_norm
from bornagain.simulate.clcore import ClCore
from bornagain.simulate.examples import psi_pdb_file, lysozyme_pdb_file
from bornagain.target import crystal, density
from bornagain.viewers.qtviews import Scatter3D, bright_colors, colors
from bornagain.external.pyqtgraph.extras import keep_open

import scipy.constants as const
eV = const.value('electron volt')
r_e = const.value("classical electron radius")

# Load up the pdb file for PSI
cryst = crystal.CrystalStructure(psi_pdb_file)
spacegroup = cryst.spacegroup
unitcell = cryst.unitcell
print(unitcell)

# Redefine operators so that PSI trimers are tightly packed
spacegroup.sym_translations[1] += np.array([1, 0, 0])
spacegroup.sym_translations[2] += np.array([1, 1, 0])
spacegroup.sym_translations[3] += np.array([1, 1, 0])
spacegroup.sym_translations[4] += np.array([0, 1, 0])
# spacegroup.sym_translations[5] += np.array([0, 1, 0])

# Setup beam and detector
beam = source.Beam(photon_energy=1800*eV, pulse_energy=1e-3, diameter_fwhm=1e-6)
pad = detector.PADGeometry(pixel_size=300e-6, distance=0.1, n_pixels=500)
q_vecs = pad.q_vecs(beam=beam).copy()
q_max = np.max(pad.q_mags(beam=beam))
resolution = 2*np.pi/q_max
mask = pad.beamstop_mask(beam=beam, q_min=2*np.pi/300e-10)
V = pad.position_vecs()
SA = pad.solid_angles()
P = pad.polarization_factors(beam=beam)
pg.image(pad.reshape(vec_mag(vec_norm(V))))
keep_open()

J = beam.photon_number_fluence
scale = pad.reshape(r_e ** 2 * SA * P * J)
print('Resolution: %.3g scale' % (resolution*1e10,))

# Density map for merging
dmap = crystal.CrystalDensityMap(cryst=cryst, resolution=resolution, oversampling=20)
dens = np.zeros(dmap.shape)
denswt = np.zeros(dmap.shape)
# Atomic scattering factors
f = cryst.molecule.get_scattering_factors(beam=beam)

# Coordinates of asymmetric unit in crystal basis x
au_x_vecs = cryst.fractional_coordinates #np.dot(cryst.molecule.coordinates, unitcell.o_mat_inv.T)
au_x_com = np.mean(au_x_vecs, axis=0)  # approximate center of mass

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
mol_x_coms = np.vstack(mol_x_coms)

# Unit cell packing
show_packing = False
if show_packing:
    scat = Scatter3D()
    for i in range(spacegroup.n_molecules):
        scat.add_points(np.dot(spacegroup.apply_symmetry_operation(i, au_x_vecs), unitcell.o_mat.T),
                        color=bright_colors(i, alpha=0.5), size=1)
    scat.add_rgb_axis(length=100e-10)
    scat.show()

# Construct the finite lattice generators
max_size = 41  # Make sure we have an odd value... for making hexagonal prisms
lats = [crystal.FiniteLattice(max_size=max_size, unitcell=unitcell) for i in range(spacegroup.n_molecules)]
# Construct a finite lattice in the form of a hexagonal prism
width = 3
length = 6
for i in range(spacegroup.n_molecules):
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

# Create a 3D scatterplot showing the crystal
viewcrystal = False
if viewcrystal:
    scat = Scatter3D()
    for i in range(spacegroup.n_molecules):
        x_vecs = lats[i].occupied_x_coordinates + mol_x_coms[i, :]
        scat.add_points(np.dot(x_vecs, unitcell.o_mat.T), color=bright_colors(i, alpha=0.5), size=5)
        sym_vecs = spacegroup.apply_symmetry_operation(i, au_x_vecs[0::1000, :])
        for j in range(x_vecs.shape[0]):
            lat_vec = lats[i].occupied_x_coordinates[j, :]
            r_vecs = np.dot(sym_vecs - (mean_com - lat_vec), unitcell.o_mat.T)
            scat.add_points(r_vecs, color=bright_colors(i, alpha=0.1), size=5)
    scat.add_rgb_axis(length=100e-10)
    scat.show()

clcore = ClCore()
print('Computing with:', clcore.context.devices)
amps_gpu = clcore.to_device(shape=pad.shape(), dtype=clcore.complex_t)*0
amps_mol_gpu = clcore.to_device(shape=pad.shape(), dtype=clcore.complex_t)
amps_lat_gpu = clcore.to_device(shape=pad.shape(), dtype=clcore.complex_t)
q_vecs_gpu = clcore.to_device(q_vecs, dtype=clcore.real_t)
q_rot_gpu = clcore.to_device(shape=q_vecs.shape, dtype=clcore.real_t)
f_gpu = clcore.to_device(f, dtype=clcore.complex_t)

for c in range(1):
    t = time()
    rot = random_rotation()
    trans = np.zeros(3)
    clcore.rotate_translate_vectors(rot, trans, q_vecs_gpu, q_rot_gpu)
    width = 2 + np.random.rand(1)*2
    length = 6 + np.random.rand(1)*2
    for i in range(spacegroup.n_molecules):
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
    amps_gpu *= 0
    for i in range(spacegroup.n_molecules):
        x = spacegroup.apply_symmetry_operation(i, au_x_vecs)
        r = unitcell.x2r(x)
        clcore.phase_factor_qrf(q_rot_gpu, lats[i].occupied_r_coordinates, a=amps_lat_gpu, add=False)
        clcore.phase_factor_qrf(q_rot_gpu, r, f_gpu, a=amps_mol_gpu, add=False)
        amps_gpu += amps_lat_gpu * amps_mol_gpu
    intensities = scale * np.abs(amps_gpu.get()) ** 2
    intensities = np.random.poisson(intensities).astype(np.float64)
    density.trilinear_insertion(densities=dens, weights=denswt, vectors=unitcell.q2h(q_rot_gpu.get()/np.pi/2.),
                                vals=intensities.ravel(), corners=dmap.h_limits[:, 0].copy(),
                                deltas=(dmap.h_limits[:, 1]-dmap.h_limits[:, 0]))
    print('Pattern %d; %.3f seconds' % (c, time()-t,))


pg.image(intensities*mask)
w = np.where(denswt > 0)[0]
d = dens.copy()
print(np.isnan(d).any(), w)
d.flat[w] /= denswt.flat[w]
pg.image(dens)
keep_open()
