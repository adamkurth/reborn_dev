r"""
Water background scatter
========================

Simple example of how to simulate water background scatter.

Contributed by Richard A. Kirian.
"""

import numpy as np
import matplotlib.pyplot as plt
import reborn
from reborn.simulate import solutions
from scipy import constants as const

np.random.seed(0)
r_e = const.value('classical electron radius')
eV = const.value('electron volt')

pad = reborn.detector.PADGeometry(distance=0.2, shape=[4000]*2, pixel_size=100e-6)
beam = reborn.source.Beam(photon_energy=8000*eV, diameter_fwhm=5e-6, pulse_energy=1e8*8000*eV)
jet_diameter = 100e-6
n_water_molecules = jet_diameter * beam.diameter_fwhm**2 * solutions.water_number_density()
qmag = pad.q_mags(beam=beam)
J = beam.fluence
P = pad.polarization_factors(beam=beam)
SA = pad.solid_angles()
F = solutions.get_water_profile(qmag, temperature=300)
F2 = F**2*n_water_molecules
I = r_e**2 * J * P * SA * F2 / beam.photon_energy
I = np.random.poisson(I)
I = pad.reshape(I)
plt.imshow(np.log10(I+1), cmap='gray', interpolation='nearest')
plt.colorbar()
plt.show()
