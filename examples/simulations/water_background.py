import sys
import numpy as np
import matplotlib.pyplot as plt
import bornagain
from bornagain.simulate import solutions
from bornagain.units import r_e, keV


pad = bornagain.detector.PADGeometry()
pad.simple_setup(distance=0.1, n_pixels=1000, pixel_size=200e-6)
beam = bornagain.source.Beam(photon_energy=9/keV, diameter_fwhm=5e-6, pulse_energy=5e-3)
jet_diameter = 3e-6
n_water_molecules = jet_diameter * beam.diameter_fwhm**2 * solutions.water_number_density()
q = pad.q_vecs(beam=beam)
qmag = bornagain.utils.vec_mag(q)
J = beam.fluence
P = pad.polarization_factors(beam=beam)
SA = pad.solid_angles()
F = solutions.get_water_profile(qmag, temperature=(25+273.16))
F2 = F**2*n_water_molecules
I = r_e**2 * J * P * SA * F2 / beam.photon_energy
I = np.random.poisson(I)
I = pad.reshape(I)

if 'noplots' not in sys.argv:

    plt.imshow(I, cmap='gray', interpolation='nearest')
    plt.colorbar()
    plt.title('water. %g µm pix.  %g m dist. %g mJ' % (pad.pixel_size()*1e6, pad.t_vec.flat[2], beam.pulse_energy*1e3))
    plt.show()
