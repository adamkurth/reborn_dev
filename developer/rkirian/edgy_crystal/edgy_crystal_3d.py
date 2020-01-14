import sys
from time import time
import numpy as np
from numpy.fft import fftn, ifftn, fftshift
from bornagain.simulate.clcore import ClCore
from bornagain.target import crystal
from bornagain.viewers.qtviews import Scatter3D, bright_colors, PADView, MapProjection, MapSlices
import scipy.constants as const
import argparse
import pyqtgraph as pg

eV = const.value('electron volt')
r_e = const.value("classical electron radius")

parser = argparse.ArgumentParser('Simulate finite crystals and merge intensities into a 3D map.')
parser.add_argument('--resolution', type=float, default=30e-10, required=False,
                    help='Minimum resolution of the density map in meters')
parser.add_argument('--n_crystals', type=int, default=10, required=False,
                    help='How many crystals to simulate (and merge)')
parser.add_argument('--oversampling', type=int, default=10, required=False,
                    help='Oversampling factor (1 corresponds to ordinary "Bragg sampling")')
parser.add_argument('--photon_energy_ev', type=float, default=8000, required=False,
                    help='Photon energy in electron volts')
parser.add_argument('--save_results', action='store_false', required=False,
                    help='Save the density map')  # 'store_false' means True by default
parser.add_argument('--checkpoint_save_interval', type=int, default=1e10, required=False, 
                    help='Save curent results at this interval')
parser.add_argument('--view_crystal', action='store_false', required=False,
                    help='Display the crystal molecule coordinates')  # 'store_false' means True by default
parser.add_argument('--view_density', action='store_false', required=False,
                    help='Display the density map in crystal coordinates')  # 'store_false' means True by default
parser.add_argument('--view_intensities', action='store_false', required=False,
                    help='Display the final averaged intensities')  # 'store_false' means True by default
parser.add_argument('--add_facets', action='store_false', required=False,
                    help='Add facets to the crystals')   # This means True by default
parser.add_argument('--run_number', type=int, default=1, required=False,
                    help='Files will be prefixed with "run%%04d_"')
parser.add_argument('--pdb_file', type=str, default='1jb0', required=False,
                    help='PDB file path or ID')
args = parser.parse_args()

photon_energy = args.photon_energy_ev * eV

cryst = crystal.CrystalStructure(args.pdb_file, tight_packing=True)  # Tight packing: put molecule COMs inside unit cell

print('# atoms:', cryst.fractional_coordinates.shape[0])
# Center of mass coordinates for the symmetry partner molecules
au_x_coms = [cryst.spacegroup.apply_symmetry_operation(i, cryst.fractional_coordinates_com)
              for i in range(cryst.spacegroup.n_operations)]

# Atomic scattering factors
f = cryst.molecule.get_scattering_factors(photon_energy=photon_energy)

# Construct the finite lattice generators
max_size = 41
lats = [crystal.FiniteLattice(max_size=max_size, unitcell=cryst.unitcell) for i in range(cryst.spacegroup.n_molecules)]

# Density map parameters
cdmap = crystal.CrystalDensityMap(cryst=cryst, resolution=args.resolution, oversampling=args.oversampling)
print('Density map shape:', cdmap.shape)

clcore = ClCore()

mol_amps = []  # Molecular transform amplitudes, 3D arrays one for each symmetry partner
for k in range(cryst.spacegroup.n_operations):
    x = cryst.spacegroup.apply_symmetry_operation(k, cryst.fractional_coordinates)
    lattice_amps = clcore.to_device(shape=cdmap.shape, dtype=clcore.complex_t) * 0
    clcore.phase_factor_mesh(x, f, N=cdmap.shape, q_min=cdmap.h_limits[:, 0]*2*np.pi,
                             q_max=cdmap.h_limits[:, 1]*2*np.pi, a=lattice_amps, add=True)
    mol_amps.append(lattice_amps)

if args.view_density:
    # Show the asymmetric unit:
    lattice_amps = mol_amps[0].get()
    lattice_amps = np.reshape(lattice_amps, cdmap.shape)
    mp = MapProjection(np.abs(fftn(lattice_amps)), title='Asymmetric unit')

if args.view_density:
    # Show the unit cell:
    cell_amps = 0
    for k in range(cryst.spacegroup.n_operations):
        cell_amps += mol_amps[k]
    cell_amps = cell_amps.get()  # Transfer off of GPU
    cell_amps = np.reshape(cell_amps, cdmap.shape)
    mp = MapProjection(np.abs(fftn(cell_amps)))

lattice_amps = clcore.to_device(shape=cdmap.shape, dtype=clcore.complex_t)
intensity_sum = clcore.to_device(shape=cdmap.shape, dtype=clcore.real_t) * 0

# For viewing the crystal lattice and molecule coordinates
if args.view_crystal:
    scat = Scatter3D()

for c in range(args.n_crystals):
    sys.stdout.write('Simulating crystal %d... ' % (c,))
    t = time()
    # Construct a finite lattice in the form of a hexagonal prism
    width = 1 #+ np.random.rand(1)*1  # todo: fix this line
    length = 1 #+ np.random.rand(1)*1  # todo: fix this line
    if args.add_facets:
        for k in range(len(lats)):
            lats[k].make_hexagonal_prism(width=width, length=length, shift=au_x_coms[k])
            lats[k].make_parallelepiped(shape=(3, 3, 3), shift=au_x_coms[k])  # todo: fix this line
    crystal_amps = 0
    for k in range(cryst.spacegroup.n_molecules):
        x = lats[k].occupied_x_coordinates
        # x = np.zeros((2, 3))  # todo: fix this line
        # x[0, 0] = 2  # todo: fix this line
        # x[1, 0] = 3  # todo: fix this line
        # print(x)
        lattice_amps *= 0
        clcore.phase_factor_mesh(x, N=cdmap.shape, q_min=cdmap.h_limits[:, 0] * 2 * np.pi,
                                 q_max=cdmap.h_limits[:, 1] * 2 * np.pi, a=lattice_amps, add=False)
        crystal_amps += lattice_amps  # * mol_amps[k]  # todo: fix this line
        if args.view_crystal:
            print(au_x_coms[k])  # todo: fix this line
            r = cryst.unitcell.x2r(x + au_x_coms[k])
            scat.add_points(r, color=bright_colors(k, alpha=0.5), size=5)
            r = cryst.unitcell.x2r(cryst.spacegroup.apply_symmetry_operation(k, cryst.fractional_coordinates))
            scat.add_points(r, color=bright_colors(k, alpha=0.5), size=1)
        break # todo: fix this line
    if args.view_crystal:
        scat.add_rgb_axis()
        scat.add_unit_cell(cell=cryst.unitcell)
        scat.set_orthographic_projection()
        scat.show()
        args.view_crystal = False
    if (((c+1) % args.checkpoint_save_interval) == 0) and args.save_results:
        filename = 'run%04d_checkpoint%06d.npz' % (args.run_number, c+1)
        sys.stdout.write('(saving %s)' % (filename,))
        np.savez(filename,
                 map=intensity_sum.get().reshape(cdmap.shape),
                 shape=cdmap.shape, representation='x',
                 map_min=np.squeeze(cdmap.h_limits[:, 0]),
                 map_max=np.squeeze(cdmap.h_limits[:, 1]))
    clcore.mod_squared_complex_to_real(crystal_amps, intensity_sum)  # Note that this operation **adds** to the intensities
    sys.stdout.write(' in %g seconds.\n' % (time() - t,))

intensity = np.reshape(intensity_sum.get(), cdmap.shape) / args.n_crystals


h_vecs = cdmap.h_vecs
ha = np.dot(h_vecs, np.array([1, 0, 0]))
hb = np.dot(h_vecs, np.array([0, 1, 0]))
hc = np.dot(h_vecs, np.array([0, 0, 1]))

intensity_model = 3*np.sin(3*2*np.pi*ha)/np.sin(2*np.pi*ha)
intensity_model *= 3*np.sin(3*2*np.pi*hb)/np.sin(2*np.pi*hb)
intensity_model *= 3*np.sin(3*2*np.pi*hc)/np.sin(2*np.pi*hc)
w = np.where(np.sin(2*np.pi*ha)*np.sin(2*np.pi*hb)*np.sin(2*np.pi*hc) == 0)
intensity_model[w] = 3**3
intensity_model *= intensity_model


print(np.max(intensity.ravel()[w] - intensity_model[w]))

if args.save_results:
    filename = 'run%04d.npz' % (args.run_number,)
    print('saving %s' % (filename,))
    np.savez('run%04d.npz' % (args.run_number,),
             map=intensity,
             shape=cdmap.shape, representation='x',
             map_min=np.squeeze(cdmap.h_limits[:, 0]),
             map_max=np.squeeze(cdmap.h_limits[:, 1]))

if args.view_intensities:
    dispim = intensity.copy()
    # dispim -= np.min(dispim)
    # dispim = dispim/np.max(dispim)
    # dispim += 1e-3
    # dispim[dispim == np.max(dispim)] = 0
    dispim = np.log(dispim)
    # thresh = np.max(dispim)*1e-10
    # dispim[dispim > thresh] = thresh
    # pg.image(np.transpose(dispim, [1, 0, 2]))
    # pg.mkQApp().exec_()
    MapSlices(dispim, title='Averaged Intensities')

print('Done!')