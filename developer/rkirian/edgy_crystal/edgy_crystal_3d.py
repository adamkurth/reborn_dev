import sys
from time import time
import numpy as np
from numpy.fft import fftn, ifftn, fftshift, ifftshift
from bornagain.simulate.clcore import ClCore
from bornagain.target import crystal
from bornagain.viewers.qtviews import Scatter3D, bright_colors, PADView, MapProjection, MapSlices, view_finite_crystal
import scipy.constants as const
import argparse

eV = const.value('electron volt')

parser = argparse.ArgumentParser('Simulate finite hexagonal prism crystals and merge intensities into a 3D map.')
parser.add_argument('--pdb_file', type=str, default='1jb0', required=False,
                    help='PDB file path or ID')
parser.add_argument('--resolution', type=float, default=10e-10, required=False,
                    help='Minimum resolution of the density map in meters')
parser.add_argument('--oversampling', type=int, default=4, required=False,
                    help='Oversampling factor (1 corresponds to ordinary "Bragg sampling")')
parser.add_argument('--n_crystals', type=int, default=10, required=False,
                    help='How many crystals to simulate (and merge)')
parser.add_argument('--crystal_length', type=str, default='6,10', required=False,
                    help='Range of crystal lengths')
parser.add_argument('--crystal_width', type=str, default='2,4', required=False,
                    help='Range of crystal widths')
parser.add_argument('--gaussian_disorder_sigmas', type=str, default='0.0,0.0,0.0', required=False,
                    help='Add Gaussian disorder with these sigmas (3 values, specified in crystal basis)')
parser.add_argument('--photon_energy_ev', type=float, default=8000, required=False,
                    help='Photon energy in electron volts')
parser.add_argument('--view_crystal', action='store_true', required=False,
                    help='Display the crystal molecule coordinates')  # 'store_true' means False by default
parser.add_argument('--view_density', action='store_true', required=False,
                    help='Display the density map in crystal coordinates')
parser.add_argument('--view_intensities', action='store_true', required=False,
                    help='Display the final averaged intensities')
parser.add_argument('--save_results', action='store_true', required=False,
                    help='Save the averaged intensities and asymmetric unit amplitudes')
parser.add_argument('--run_number', type=int, default=1, required=False,
                    help='Files will be prefixed with "run%%04d_"')
args = parser.parse_args()
args.crystal_length = [float(a) for a in args.crystal_length.split(',')]
args.crystal_width = [float(a) for a in args.crystal_width.split(',')]
args.gaussian_disorder_sigmas = [float(a) for a in args.gaussian_disorder_sigmas.split(',')]

# This has molecule, unit cell, and spacegroup
cryst = crystal.CrystalStructure(args.pdb_file, tight_packing=True)  # Tight packing: put molecule COMs inside unit cell

# This uses a crystal structure to make finite lattices with spacegroup considerations
fc = crystal.FiniteCrystal(cryst, max_size=20)
fc.set_gaussian_disorder(sigmas=args.gaussian_disorder_sigmas)

# Density map configuration with spacegroup considerations
cdmap = crystal.CrystalDensityMap(cryst=cryst, resolution=args.resolution, oversampling=args.oversampling)

# GPU simulation engine
clcore = ClCore()

# Calculate 3D molecular transform amplitudes once.
mol_amps = []
f = clcore.to_device(cryst.molecule.get_scattering_factors(photon_energy=args.photon_energy_ev*eV),
                     dtype=clcore.complex_t)
for k in range(cryst.spacegroup.n_operations):
    x = cryst.spacegroup.apply_symmetry_operation(k, cryst.fractional_coordinates)
    amp = clcore.to_device(shape=cdmap.shape, dtype=clcore.complex_t) * 0
    clcore.phase_factor_mesh(x, f, N=cdmap.shape, q_min=cdmap.h_limits[:, 0]*2*np.pi,
                             q_max=cdmap.h_limits[:, 1]*2*np.pi, a=amp, add=True)
    mol_amps.append(amp)

del f

if args.view_crystal:   # 3D view of crystal
    width = args.crystal_width[0] + np.random.rand(3)*args.crystal_width[1]
    length = args.crystal_length[0] + np.random.rand(1)*args.crystal_length[1]
    fc.make_hexagonal_prism(width=width, length=length)
    view_finite_crystal(fc)

if args.view_density:   # Show the unit cell density map
    cell_amps = 0
    for k in range(cryst.spacegroup.n_operations):
        cell_amps += mol_amps[k]
    dens = ifftn(ifftshift(cell_amps.get().reshape(cdmap.shape)))
    # dens[0:5, 0:5, 0:30] = np.max(np.abs(dens))
    # dens[0:5, 0:20, 0:5] = np.max(np.abs(dens))
    # dens[0:10, 0:5, 0:5] = np.max(np.abs(dens))
    MapProjection(np.abs(dens))

import pyqtgraph as pg
dens = ifftn(ifftshift(mol_amps[0].get().reshape(cdmap.shape)))
pg.image(np.angle(dens))
pg.mkQApp().exec_()
pg.plot(np.real(dens).ravel()[0:1000])
pg.show()
print(np.sum(np.abs(np.abs(dens)-np.real(dens)))/np.sum(np.abs(np.abs(dens))))
broke

lattice_amps = clcore.to_device(shape=cdmap.shape, dtype=clcore.complex_t)
intensity_sum = clcore.to_device(shape=cdmap.shape, dtype=clcore.real_t) * 0

for c in range(args.n_crystals):
    sys.stdout.write('Simulating crystal %d... ' % (c,))
    t = time()
    # Construct a finite lattice in the form of a hexagonal prism
    width = args.crystal_width[0] + np.random.rand(3)*args.crystal_width[1]
    length = args.crystal_length[0] + np.random.rand(1)*args.crystal_length[1]
    fc.make_hexagonal_prism(width=width, length=length)
    crystal_amps = 0
    for k in range(cryst.spacegroup.n_molecules):
        x = fc.lattices[k].occupied_x_coordinates
        clcore.phase_factor_mesh(x, N=cdmap.shape, q_min=cdmap.h_limits[:, 0] * 2 * np.pi,
                                 q_max=cdmap.h_limits[:, 1] * 2 * np.pi, a=lattice_amps, add=False)
        crystal_amps += lattice_amps * mol_amps[k]
    clcore.mod_squared_complex_to_real(crystal_amps, intensity_sum, add=True)
    sys.stdout.write(' in %g seconds.\n' % (time() - t,))

intensity = np.reshape(intensity_sum.get(), cdmap.shape) / args.n_crystals
w = np.where(~np.isfinite(intensity))
if np.array(w).squeeze().size > 0:
    print('Found non-finite values in dispim.  They will be set to -1.')
    print(intensity[w])
    intensity[w] = -1

if args.save_results:
    filename = 'run%04d_intensity.npz' % (args.run_number,)
    print('saving %s' % (filename,))
    np.savez(filename, map=intensity, type='intensity', shape=cdmap.shape, representation='h',
             map_min=np.squeeze(cdmap.h_limits[:, 0]), map_max=np.squeeze(cdmap.h_limits[:, 1]))
    filename = 'run%04d_au_amplitude.npz' % (args.run_number,)
    np.savez(filename, map=mol_amps[0].get().reshape(cdmap.shape), type='amplitude', shape=cdmap.shape,
             representation='h', map_min=np.squeeze(cdmap.h_limits[:, 0]), map_max=np.squeeze(cdmap.h_limits[:, 1]))

if args.view_intensities:
    dispim = intensity.copy()
    dispim -= np.min(dispim)
    dispim /= np.max(dispim)
    dispim *= 10000
    dispim += 1
    dispim = np.log(dispim)
    # w = np.where(~np.isfinite(dispim))
    # if w.size > 0:
    #     print('Found non-finite values in dispim')
    # dispim[w] = 0
    # print(np.isfinite(dispim).all())
    MapSlices(dispim, title='Averaged Intensities')

print('Done!')
