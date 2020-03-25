import sys
from time import time
import numpy as np
from numpy.fft import fftn, ifftn, fftshift, ifftshift
from reborn.simulate.clcore import ClCore
from reborn.target import crystal
from reborn.viewers.qtviews import Scatter3D, bright_colors, PADView, MapProjection, MapSlices, view_finite_crystal
import scipy.constants as const
import argparse

import pyqtgraph as pg

pg.setConfigOption('background', np.ones(3)*50)
pg.setConfigOption('foreground', np.ones(3)*255)

eV = const.value('electron volt')

parser = argparse.ArgumentParser('Simulate finite hexagonal prism crystals and merge intensities into a 3D map.')
parser.add_argument('--pdb_file', type=str, default='1jb0', required=False,
                    help='PDB file path or ID')
parser.add_argument('--resolution', type=float, default=10e-10, required=False,
                    help='Minimum resolution of the density map in meters')
parser.add_argument('--oversampling', type=int, default=4, required=False,
                    help='Oversampling factor (1 corresponds to ordinary "Bragg sampling")')
parser.add_argument('--direct_molecular_transform', action='store_true', required=False,
                    help='Calculate molecular transforms by direct summation of atomic scattering factors')
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

# This has molecule, unit cell, and spacegroup info
cryst = crystal.CrystalStructure(args.pdb_file, tight_packing=True)  # Tight packing: put molecule COMs inside unit cell
uc = cryst.unitcell
sg = cryst.spacegroup
print(uc)
print(sg)

# sys.exit()

# P63 Twinning operations
# for k in range(sg.n_operations):
#     v = sg.sym_translations[k]
#     r = sg.sym_rotations[k]
#     sg.sym_translations[k] = np.array([v[1], v[0], -v[2]])
#     sg.sym_rotations[k] = np.array([[r[0, 1], r[0, 0], -r[0, 2]],
#                                     [r[1, 1], r[1, 0], -r[1, 2]],
#                                     [r[2, 1], r[2, 0], -r[2, 2]]])
# cryst.set_tight_packing()

# This uses a crystal structure to make finite lattices with spacegroup considerations
fc = crystal.FiniteCrystal(cryst, max_size=20)
fc.set_gaussian_disorder(sigmas=args.gaussian_disorder_sigmas)

# Density map configuration with spacegroup considerations
cdmap = crystal.CrystalDensityMap(cryst=cryst, resolution=args.resolution, oversampling=args.oversampling)

# FIXME : Set atom coordinates to grid points for testing of the two different methods discussed below.  By setting
#         atoms to lie on gridpoints, we *should* in principle get the same results from both methods (but we don't).
# cryst.fractional_coordinates = np.floor(cryst.fractional_coordinates/cdmap.dx)*cdmap.dx

# GPU simulation engine
clcore = ClCore()

# Scattering factors (absolute value because interpolations don't work with complex numbers)
f = cryst.molecule.get_scattering_factors(photon_energy=args.photon_energy_ev*eV)

# FIXME: Which way are we supposed to do this?
# For now, we are going to compute the molecular transform amplitudes in two different ways.  The "direct" method is to
# sum over all the atoms with the usual formula: sum_n f_n exp(i q.r_n).
# The other way, the "fft" method, consists of firstly making the density map and subsequently taking the FFT of the
# resulting map.
# We can expect that the direct and fft methods will differ somewhat because one is done on a GPU (possibly with single)
# precision) whereas the other is done on a CPU with double precision.  There are multiple ways in which floating
# point arithmetic is handled and this may differ from one device, compiler, etc. to another.  There will also be
# slightly different results because the order of the summation differs in each case; on a computer, summing the
# densities first and then multiplying that sum by the exponential phase factor is different than summing over the
# density/phase factor products.  To be clear, the following test will fail:
#
#     a = np.arange(10000)
#     b = np.sum(a)*np.exp(1j*3)
#     c = np.sum(a*np.exp(1j*3))
#     assert np.sum(np.abs(c - b)) == 0

# Calculate 3D molecular transform amplitudes on GPU via explicit atomic coordinates: sum over f * exp(i q.r)
# The issue with this is that we get ringing artifacts when we take FFTs, but this is more like real data.
mol_amps_direct = []
f_gpu = clcore.to_device(f, dtype=clcore.complex_t)
for k in range(cryst.spacegroup.n_operations):
    x = cryst.spacegroup.apply_symmetry_operation(k, cryst.fractional_coordinates)
    amp = clcore.to_device(shape=cdmap.shape, dtype=clcore.complex_t) * 0
    clcore.phase_factor_mesh(x, f_gpu, N=cdmap.shape, q_min=cdmap.h_min, q_max=cdmap.h_max, a=amp, add=False, twopi=True)
    mol_amps_direct.append(amp)

# Build the electron densities directly and make amplitudes via FFT.  This avoids ringing artifacts that we get
# from the direct summation method.
mol_amps_fft = []
print('sum over f', np.sum(f))
au_map = cdmap.place_atoms_in_map(cryst.fractional_coordinates, f, mode='gaussian', fixed_atom_sigma=10e-10)
print('sum over au_map', np.sum(au_map))
for k in range(cryst.spacegroup.n_operations):
    rho = cdmap.au_to_k(k, au_map)
    mol_amps_fft.append(clcore.to_device(fftshift(fftn(rho)), dtype=clcore.complex_t))
print('k zero direct', ifftshift(mol_amps_direct[0].get())[0, 0, 0])
print('k zero fft', ifftshift(mol_amps_fft[0].get())[0, 0, 0])

# Here we choose which of the above methods will go into the saved results:
if args.direct_molecular_transform:
    mol_amps = mol_amps_direct
else:
    mol_amps = mol_amps_fft

# Check how much the two methods differ:
for k in range(cryst.spacegroup.n_operations):
    a = np.abs(mol_amps_direct[k].get()).ravel()
    b = np.abs(mol_amps_fft[k].get()).ravel()
    print('symmetry partner', k)
    print('k zero direct', ifftshift(mol_amps_direct[k].get())[0, 0, 0])
    print('k zero fft', ifftshift(mol_amps_fft[k].get())[0, 0, 0])
    print('k zero direct', ifftshift(mol_amps_direct[k].get())[5, 5, 5])
    print('k zero fft', ifftshift(mol_amps_fft[k].get())[5, 5, 5])
    print('Difference in abs(amplitudes):', np.sum(np.abs(a-b))/(np.sum(np.abs(a+b))/2))

levels = [0, 500]
a = np.abs(mol_amps_direct[0].get())
imwin_direct = pg.image(a, title='direct')
imwin_direct.setLevels(levels[0], levels[1])
imwin_direct.setCurrentIndex(int(np.ceil(a.shape[0]/2)))
imwin_direct.setPredefinedGradient('flame')
b = np.abs(mol_amps_fft[0].get())
imwin_fft = pg.image(b, title='fft')
imwin_fft.setLevels(levels[0], levels[1])
imwin_fft.setCurrentIndex(int(np.ceil(b.shape[0]/2)))
imwin_fft.setPredefinedGradient('flame')
imwin_diff = pg.image(b-a, title='fft - direct')
imwin_fft.setCurrentIndex(int(np.ceil(b.shape[0]/2)))
imwin_fft.setPredefinedGradient('flame')


if args.view_crystal:   # 3D view of crystal
    width = args.crystal_width[0] + np.random.rand(3)*args.crystal_width[1]
    length = args.crystal_length[0] + np.random.rand(1)*args.crystal_length[1]
    fc.make_hexagonal_prism(width=width, length=length)
    view_finite_crystal(fc)

if args.view_density:   # Show the unit cell density map

    dens = ifftn(ifftshift(mol_amps[0].get().reshape(cdmap.shape)))
    MapProjection(np.abs(dens), title='Asymmetric Unit')

    cell_amps = 0
    for k in range(cryst.spacegroup.n_operations):
        cell_amps += mol_amps[k]
    dens = ifftn(ifftshift(cell_amps.get().reshape(cdmap.shape)))
    MapProjection(np.abs(dens), title='Unit Cell')

lattice_amps = clcore.to_device(shape=cdmap.shape, dtype=clcore.complex_t)
intensity_sum = clcore.to_device(shape=cdmap.shape, dtype=clcore.real_t) * 0

for c in range(args.n_crystals):
    sys.stdout.write('Simulating crystal %d... ' % (c,))
    t = time()
    # Construct a finite lattice in the form of a hexagonal prism
    width = args.crystal_width[0] + np.random.rand(3)*(args.crystal_width[1]-args.crystal_width[0])
    length = args.crystal_length[0] + np.random.rand(1)*(args.crystal_length[1]-args.crystal_length[0])
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

if args.save_results:
    filename = 'run%04d_intensity.npz' % (args.run_number,)
    print('saving %s' % (filename,))
    np.savez(filename, map=intensity, type='intensity', shape=cdmap.shape, representation='h',
             map_min=np.squeeze(cdmap.h_limits[:, 0]), map_max=np.squeeze(cdmap.h_limits[:, 1]))
    filename = 'run%04d_au_amplitude.npz' % (args.run_number,)
    print('saving %s' % (filename,))
    np.savez(filename, map=mol_amps[0].get().reshape(cdmap.shape), type='amplitude', shape=cdmap.shape,
             representation='h', map_min=np.squeeze(cdmap.h_limits[:, 0]), map_max=np.squeeze(cdmap.h_limits[:, 1]))

if args.view_intensities:
    dispim = intensity.copy()
    dispim -= np.min(dispim)
    dispim /= np.max(dispim)
    dispim *= 10000
    dispim += 1
    dispim = np.log(dispim)
    MapSlices(dispim, title='Averaged Intensities')

app = pg.mkQApp()
app.exec_()

print('Done!')
