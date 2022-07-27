import numpy as np
from numpy.fft import fftn, ifftn, fftshift, ifftshift
import pyqtgraph as pg
from reborn import const, source
from reborn.simulate.clcore import ClCore
from reborn.simulate.examples import psi_pdb_file, lysozyme_pdb_file
from reborn.target import crystal
from reborn.external.pyqtgraph import keep_open
eV = const.eV
r_e = const.r_e
beam = source.Beam(photon_energy=5000*eV, pulse_energy=1e-3, diameter_fwhm=1e-6)

# Load up the pdb file
cryst = crystal.CrystalStructure(psi_pdb_file)
cryst.spacegroup.sym_rotations = cryst.spacegroup.sym_rotations[:]  # TODO: fix this
cryst.spacegroup.sym_translations = cryst.spacegroup.sym_translations[:]  # TODO: fix this
cryst.fractional_coordinates = cryst.fractional_coordinates[:] # np.array([[0.4, 0, 0], [0.5, 0, 0]])  # TODO: fix this
r_vecs = cryst.unitcell.x2r(cryst.fractional_coordinates)
f = cryst.molecule.get_scattering_factors(beam=beam)

# 3D merge map
oversampling = 4
h_max = 15
h_min = -h_max
n_h_bins = (oversampling*(h_max-h_min)*np.ones([3])+1).astype(int)
h_corner_min = h_min*np.ones([3])
h_corner_max = h_max*np.ones([3])
h1d = np.arange(h_min, h_max, (h_max-h_min)/(n_h_bins[0]-1))
hx, hy, hz = np.meshgrid(h1d, h1d, h1d, indexing='ij')
h_vecs = np.vstack((hx.ravel(), hy.ravel(), hz.ravel())).T.copy()
h_mags = np.sqrt(np.sum(h_vecs**2, axis=1))
gaus = np.exp(-h_mags**2/((h_max/2)**2)).reshape(n_h_bins-1)
q_vecs = cryst.unitcell.h2q(h_vecs)

# print('h_vecs:')
# print(h_vecs)

clcore = ClCore()
# print('Computing with:', clcore.context.devices)
amps = clcore.phase_factor_qrf(q_vecs, r_vecs, f)
amps = amps.reshape(n_h_bins-1)
intensities = np.abs(amps) ** 2
rho = fftn(ifftshift(amps*gaus))

rhoabs = np.abs(ifftshift(rho))
im1 = pg.image(rhoabs/np.max(rhoabs))

im2 = pg.image(intensities/1e16)
# print('Keeping open...')
keep_open()
