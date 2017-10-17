import os
import sys
import time
import numpy as np
import h5py
from glob import glob
import matplotlib.pyplot as plt

sys.path.append("../..")
import bornagain as ba
from bornagain.units import r_e, hc, keV
from bornagain.simulate.clcore import ClCore


n_molecules = 10
photon_energy = 6.5 / keV
wavelength = hc/photon_energy

beam_vec = np.array([0, 0, 1.0])

# Single molecule atomic positions:
r = np.array([[0,     0, 0],
              [5e-10, 0, 0]])
n_atoms = r.shape[0]
f = ba.simulate.atoms.get_scattering_factors([25]*r.shape[0], photon_energy)  # This number is irrelevant

pad = ba.detector.PADGeometry()
pad.simple_setup(n_pixels=100, pixel_size=100e-9, distance=1.0)
q = pad.q_vecs(beam_vec=beam_vec, wavelength=wavelength)

clcore = ClCore(group_size=32)


# For one shot, jiggle molecule positions and orientations and phases
rs = np.zeros((n_molecules*n_atoms,3))
fs = np.zeros((n_molecules*n_atoms), dtype=clcore.complex_t)

for n in range(0,n_molecules):

    # Rotate one molecule
    R = ba.utils.random_rotation()
    # Add the rotated postions to list of atoms
    rs[(n*n_atoms):((n+1)*n_atoms), :] = np.dot(R, r.T).T

    # No translation yet...

    # As a hack we will randomize the structure factor phases
    phases = np.random.random(f.shape)*2.0*np.pi
    fs[(n*n_atoms):((n+1)*n_atoms)] = f*np.exp(1j*phases)


A = clcore.phase_factor_pad(rs, fs, pad.t_vec, pad.fs_vec, pad.ss_vec, beam_vec, pad.n_fs, pad.n_ss, wavelength)
I = np.abs(A)**2  # As a practical limit, this intensity should reflect the fact that we get only one fluorescence
                # photon per atom

# Next: repeat many times, make lots of patterns, then take two-point correlations

# For incident fluence, we get the cross section (area) for photoabsorption.  We want one incdient photon per that area.

# Questions:
# How does SNR scale with number of molecules?
# How many shots needed?


# Something is wrong with this output!
imdisp = I.reshape(pad.shape())
plt.imshow(np.log10(imdisp+1e-5))
plt.show()