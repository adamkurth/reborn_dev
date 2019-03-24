import sys
sys.path.append('../..')

import numpy as np
import pyqtgraph as pg
from bornagain import detector
from bornagain import source
from bornagain import utils
from bornagain.simulate.clcore import ClCore
from bornagain.simulate.examples import lysozyme_pdb_file, psi_pdb_file
from bornagain.target import crystal
from bornagain.viewers.qtviews import Scatter3D, bright_colors, colors
from bornagain.external.pyqtgraph.extras import keep_open


cryst = crystal.CrystalStructure(psi_pdb_file)
spacegroup = cryst.spacegroup
spacegroup.sym_translations[2] += np.array([1, 0, 0])
spacegroup.sym_translations[4] += np.array([1, 1, 0])
spacegroup.sym_translations[3] += np.array([1, 1, 0])
spacegroup.sym_translations[5] += np.array([0, 1, 0])
unitcell = cryst.unitcell
mol_x_coords = utils.rotate(unitcell.o_mat_inv, cryst.molecule.coordinates)
mol_x_com = np.mean(mol_x_coords, axis=0)
max_size = 7
lat = crystal.FiniteLattice(max_size=max_size, unitcell=unitcell)
cryst_length = 0.5
cryst_width = 3
lat.make_hexagonal_prism(n_cells=cryst_width)
lat.add_facet(plane=[0, 0, 1], length=cryst_length)
lat.add_facet(plane=[0, 0, -1], length=cryst_length)
lat_x_coords = lat.occupied_x_coordinates
lat_r_coords = lat.occupied_r_coordinates

scat = Scatter3D()

for lt in range(lat_x_coords.shape[0]):

    trans = lat_x_coords[lt, :]

    for op in range(spacegroup.n_molecules):

        # xx = spacegroup.apply_symmetry_operation(op, mol_x_coords) + trans
        # r_vecs = utils.rotate(unitcell.o_mat, xx)
        # scat.add_points(r_vecs, color=bright_colors(op), size=1)
        xx = spacegroup.apply_symmetry_operation(op, mol_x_com) + trans
        r_vecs = utils.rotate(unitcell.o_mat, xx)
        scat.add_points(r_vecs, color=bright_colors(op), size=2)

# scat.add_points(utils.rotate(unitcell.o_mat, lat.all_x_coordinates), color=(0.2, 0.2, 0.2, 0.2), size=5)
# scat.add_points(utils.rotate(unitcell.o_mat, lat.occupied_x_coordinates), size=10)

scat.add_rgb_axis(length=100e-10)
scat.show()

beam = source.Beam(wavelength=3e-10)
pad = detector.PADGeometry(pixel_size=100e-6, distance=0.5, n_pixels=1000)
q_vecs = pad.q_vecs(beam=beam)
clcore = ClCore()
A = clcore.phase_factor_qrf(q_vecs, lat_r_coords, np.ones(lat_r_coords.shape[0]))
I = pad.reshape(np.abs(A)**2)
pg.image(I)
keep_open()
