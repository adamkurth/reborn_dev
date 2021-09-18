import numpy as np
import matplotlib.pyplot as plt
from reborn import detector, source
from reborn.simulate import solutions
print(detector.__file__)
pad = detector.PADGeometry(shape=(1000, 1000), distance=0.1, pixel_size=200e-6)
beam = source.Beam(wavelength=1.5e-10)
q = pad.q_mags(beam=beam)
q = np.linspace(0, 10e10, 10000)

h1 = np.genfromtxt('clark2010_25.csv', delimiter=',')
print(h1)
I = solutions.water_scattering_factor_squared(q, temperature=298, volume=None)
plt.plot(q, I)
plt.plot(h1[:, 0]*1e10, h1[:, 1]*(10)**2, '.')
plt.show()
