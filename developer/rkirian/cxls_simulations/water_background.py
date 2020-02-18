import sys
import numpy as np
import matplotlib.pyplot as plt
import bornagain
from bornagain.simulate import solutions
from scipy import constants as const

r_e = const.value('classical electron radius')
eV = const.value('electron volt')

pad = bornagain.detector.PADGeometry(distance=0.1, shape=[2000]*2, pixel_size=75e-6)
beam = bornagain.source.Beam(photon_energy=8000*eV, diameter_fwhm=5e-6, pulse_energy=1e8*8000*eV)
jet_diameter = 100e-6
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
