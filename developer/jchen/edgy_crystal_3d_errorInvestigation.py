import sys
from time import time
import numpy as np
from numpy.fft import fftn, ifftn, fftshift, ifftshift
from bornagain.simulate.clcore import ClCore
from bornagain.target import crystal
import scipy.constants as const
import argparse

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.close('all')
def colorbar(ax, im):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(im, cax=cax)


eV = const.value('electron volt')

parser = argparse.ArgumentParser('Simulate finite hexagonal prism crystals and merge intensities into a 3D map.')
parser.add_argument('--pdb_file', type=str, default='4et8', required=False,
                    help='PDB file path or ID')
parser.add_argument('--resolution', type=float, default=2e-10, required=False,
                    help='Minimum resolution of the density map in meters')
parser.add_argument('--oversampling', type=int, default=4, required=False,
                    help='Oversampling factor (1 corresponds to ordinary "Bragg sampling")')
parser.add_argument('--direct_molecular_transform', action='store_true', required=False,
                    help='Calculate molecular transforms by direct summation of atomic scattering factors')
parser.add_argument('--n_crystals', type=int, default=1, required=False,
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

# Density map configuration with spacegroup considerations
cdmap = crystal.CrystalDensityMap(cryst=cryst, resolution=args.resolution, oversampling=args.oversampling)



N_keep = 2

cryst.fractional_coordinates = np.floor(cryst.fractional_coordinates/cdmap.dx)*cdmap.dx
cryst.fractional_coordinates = cryst.fractional_coordinates[0:N_keep,:]

cryst.fractional_coordinates[1,:] += 0.1
print(cryst.fractional_coordinates)
cryst.fractional_coordinates[0,:] += 0.01
print(cryst.fractional_coordinates)

# Scattering factors (absolute value because interpolations don't work with complex numbers)
f = cryst.molecule.get_scattering_factors(photon_energy=args.photon_energy_ev*eV)
f = f[0:N_keep]
f[0] = 1
f[1] = 1

# GPU simulation engine
clcore = ClCore()


# Direct calculation
mol_amps_direct = []
f_gpu = clcore.to_device(f, dtype=clcore.complex_t)
for k in range(1):
    x = cryst.fractional_coordinates
    amp = clcore.to_device(shape=cdmap.shape, dtype=clcore.complex_t) * 0
    clcore.phase_factor_mesh(x, f_gpu, N=cdmap.shape, q_min=cdmap.h_min, q_max=cdmap.h_max, a=amp, add=False, twopi=True)
    mol_amps_direct.append(amp)

F1 = ifftshift(mol_amps_direct[0].get())



# Trilinear interpolate and then take the FFT
au_map = cdmap.place_atoms_in_map(cryst.fractional_coordinates, f, mode='trilinear')
F2 = (fftn(au_map))

print('sum over f')
print(np.sum(f))

print('sum over au_map')
print(np.sum(au_map))

print('[0,0,0] of F1')
print(F1[0,0,0])

print(np.sqrt(np.sum(np.abs(F2-F1)**2) / np.sum(np.abs(F1)**2)))

# Interpolated atom
print(au_map[4:8,0:3,3:7])


# Visualise results
clim_max = 2
clim_min = 0

fig = plt.figure()
ax = fig.add_subplot(231)
im = ax.imshow(fftshift((np.abs(F1[0,:,:]))), clim=[clim_min,clim_max], interpolation='nearest')
colorbar(ax, im)
ax.set_title('Direct qx,qy')
ax = fig.add_subplot(232)
im = ax.imshow(fftshift((np.abs(F1[:,0,:]))), clim=[clim_min,clim_max], interpolation='nearest')
colorbar(ax, im)
ax.set_title('Direct qx,qz')
ax = fig.add_subplot(233)
im = ax.imshow(fftshift((np.abs(F1[:,:,0]))), clim=[clim_min,clim_max], interpolation='nearest')
colorbar(ax, im)
ax.set_title('Direct qy,qz')

ax = fig.add_subplot(234)
im = ax.imshow(fftshift((np.abs(F2[0,:,:]))), clim=[clim_min,clim_max], interpolation='nearest')
colorbar(ax, im)
ax.set_title('TrilinearIntp+FFT qx,qy')
ax = fig.add_subplot(235)
im = ax.imshow(fftshift((np.abs(F2[:,0,:]))), clim=[clim_min,clim_max], interpolation='nearest')
colorbar(ax, im)
ax.set_title('TrilinearIntp+FFT qx,qz')
ax = fig.add_subplot(236)
im = ax.imshow(fftshift((np.abs(F2[:,:,0]))), clim=[clim_min,clim_max], interpolation='nearest')
colorbar(ax, im)
ax.set_title('TrilinearIntp+FFT qy,qz')

# plt.tight_layout()
plt.show()


fig = plt.figure()
ax = fig.add_subplot(131)
im = ax.imshow(np.abs(np.sum(au_map, axis=0)), interpolation='nearest')
colorbar(ax, im)
ax.set_title('np.sum(au_map, axis=0)')
ax = fig.add_subplot(132)
im = ax.imshow(np.abs(np.sum(au_map, axis=1)), interpolation='nearest')
colorbar(ax, im)
ax.set_title('np.sum(au_map, axis=1)')
ax = fig.add_subplot(133)
im = ax.imshow(np.abs(np.sum(au_map, axis=2)), interpolation='nearest')
colorbar(ax, im)
ax.set_title('np.sum(au_map, axis=2)')

# plt.tight_layout()
plt.show()








