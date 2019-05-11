from time import time
import numpy as np
import pyqtgraph as pg
import bornagain as ba
from bornagain.units import keV
from bornagain import detector
from bornagain import source
from bornagain.utils import rotate, random_rotation
from bornagain.simulate.clcore import ClCore
from bornagain.simulate.examples import psi_pdb_file, lysozyme_pdb_file
from bornagain.target import crystal, density
from bornagain.viewers.qtviews import Scatter3D, bright_colors, colors
from bornagain.external.pyqtgraph.extras import keep_open


# Load up the pdb file for PSI
cryst = crystal.CrystalStructure(psi_pdb_file)
spacegroup = cryst.spacegroup
unitcell = cryst.unitcell
print(unitcell)

# Redefine operators so that PSI trimers are tightly packed
spacegroup.sym_translations[2] += np.array([1, 0, 0])
spacegroup.sym_translations[4] += np.array([1, 1, 0])
spacegroup.sym_translations[3] += np.array([1, 1, 0])
spacegroup.sym_translations[5] += np.array([0, 1, 0])

# Setup beam and detector
beam = source.Beam(photon_energy=1.8/keV, pulse_energy=1e-3, diameter_fwhm=1e-6)
pad = detector.PADGeometry(pixel_size=300e-6, distance=0.5, n_pixels=500)
q_vecs = pad.q_vecs(beam=beam).copy()
q_max = np.max(pad.q_mags(beam=beam))
resolution = 2*np.pi/q_max
mask = pad.beamstop_mask(beam=beam, q_min=2*np.pi/300e-10)
print('Resolution: %.3g A' % (resolution*1e10,))

# Density map for merging
dmap = density.CrystalDensityMap(cryst=cryst, resolution=resolution, oversampling=4)
dens = dmap.zeros()
denswt = dmap.zeros()

# Atomic scattering factors
f = cryst.molecule.get_scattering_factors(beam=beam)

# Coordinates of asymmetric unit in crystal basis x
au_x_vecs = rotate(unitcell.o_mat_inv, cryst.molecule.coordinates)
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

# Construct the finite lattice generators
max_size = 41  # Make sure we have an odd value... for making hexagonal prisms
lats = [crystal.FiniteLattice(max_size=max_size, unitcell=unitcell) for i in range(spacegroup.n_molecules)]
# Construct a finite lattice in the form of a hexagonal prism
width = 3
length = 6
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
        sym_vecs = spacegroup.apply_symmetry_operation(i, au_x_vecs[0::1000, :])
        for j in range(x_vecs.shape[0]):
            lat_vec = lats[i].occupied_x_coordinates[j, :]
            r_vecs = rotate(unitcell.o_mat, sym_vecs - (mean_com + lat_vec))
            scat.add_points(r_vecs, color=bright_colors(i, alpha=0.1), size=1)
    scat.add_rgb_axis(length=100e-10)
    scat.show()

clcore = ClCore()
print('Computing with:', clcore.context.devices)
amps_gpu = clcore.to_device(shape=pad.shape(), dtype=clcore.complex_t)*0
amps_mol_gpu = clcore.to_device(shape=pad.shape(), dtype=clcore.complex_t)
amps_lat_gpu = clcore.to_device(shape=pad.shape(), dtype=clcore.complex_t)
q_vecs_gpu = clcore.to_device(q_vecs, dtype=clcore.real_t)
q_rot_gpu = clcore.to_device(shape=q_vecs.shape, dtype=clcore.real_t)

for c in range(10):
    print(c)
    t = time()
    rot = random_rotation()
    trans = np.zeros(3)
    clcore.rotate_translate_vectors(rot, trans, q_vecs_gpu, q_rot_gpu)
    amps_gpu *= 0
    for i in range(1):#spacegroup.n_molecules):
        x = spacegroup.apply_symmetry_operation(i, au_x_vecs)
        r = unitcell.x2r(x)
        clcore.phase_factor_qrf(q_rot_gpu, lats[i].occupied_r_coordinates, a=amps_lat_gpu, add=False)
        clcore.phase_factor_qrf(q_rot_gpu, r, a=amps_mol_gpu, add=False)
        amps_gpu += amps_lat_gpu #* amps_mol_gpu

    intensities = (np.abs(amps_gpu.get())**2).astype(np.float64)
    print('GPU simulation: %g seconds' % (time()-t,))
    density.trilinear_insertion(densities=dens, weights=denswt, vectors=unitcell.q2h(rotate(rot, q_vecs)),
                                vals=intensities.ravel(), corners=dmap.h_corner, deltas=dmap.h_deltas)

pg.image(intensities*mask)
w = np.where(denswt > 0)[0]
d = dens.copy()
d.flat[w] /= denswt.flat[w]
pg.image(dens)
keep_open()
