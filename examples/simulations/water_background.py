import sys
sys.path.append('../..')


import numpy as np
import matplotlib.pyplot as plt
import bornagain as ba
from bornagain.simulate import solutions
from bornagain.units import r_e


distance = 0.1
n_pixels = 1000
wavelength = 1e-10
beam_vec = np.array([0, 0, 1])
polarization_vec = np.array([1, 0, 0])
water_number_density = 33.3679e27
beam_area = (1e-6)**2
photons_per_pulse = 1e12
J = photons_per_pulse/beam_area
water_thickness = 2e-6
n_water_molecules = beam_area*water_thickness*water_number_density

pad = ba.detector.PADGeometry()
pad.simple_setup(n_pixels=n_pixels, pixel_size=100e-6, distance=distance)

q = pad.q_vecs(beam_vec=beam_vec, wavelength=wavelength)
qmag = ba.utils.vec_mag(q)
# print(np.max(qmag), np.min(qmag))
P = pad.polarization_factors(beam_vec=beam_vec, polarization_vec=polarization_vec)
SA = pad.solid_angles()

F = solutions.get_water_profile(qmag, temperature=(25+273.16))
F2 = F**2*n_water_molecules
I = J * r_e**2 * P * SA * F2
I = np.random.poisson(I)
I = pad.reshape(I)
plt.imshow(I, cmap='gray', interpolation='nearest')
plt.show()

qmag = np.arange(0, 4e10, 4e10/1000.)
F = solutions.get_water_profile(qmag, temperature=(20+273.16))
I1 = J * r_e**2 * F**2 * n_water_molecules
F = solutions.get_water_profile(qmag, temperature=(40+273.16))
I2 = J * r_e**2 * F**2 * n_water_molecules
plt.plot(qmag, I2-I1, '.')
plt.show()
plt.title('Difference profile (two temperatures)')
