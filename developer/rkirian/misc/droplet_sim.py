import numpy as np
import xraylib
import matplotlib.pyplot as plt
import xraylib
import pyqtgraph as pg
from reborn.detector import tiled_pad_geometry_list
from reborn.source import Beam
from reborn.target.crystal import FiniteLattice, UnitCell, CrystalStructure, pdb_to_dict
from reborn.viewers.qtviews import PADView, Scatter3D
from reborn.simulate.atoms import xraylib_scattering_factors, atomic_symbols_to_numbers
from reborn.simulate.clcore import ClCore
from reborn.simulate.form_factors import sphere_form_factor
from reborn.utils import max_pair_distance
import scipy.constants as const

eV = const.value('electron volt')
r_e = const.value('classical electron radius')
NA = const.value('Avogadro constant')
h = const.h
c = const.c

water_density = 1000
photon_energy = 2000*eV
wavelength = h*c/photon_energy
distance = .5
pulse_energy = 1e-3
drop_radius = 20e-9

beam = Beam(photon_energy=photon_energy, diameter_fwhm=1e-6, pulse_energy=pulse_energy)
fluence = beam.photon_number_fluence

# Construct the CXI Jungfrau 4M detector, made up of 8 modules arranged around a 9mm beamhole.  The number of pixels per
# module is 1024 x 512 and the pixel size is 75 microns.
bin = 1
pads = tiled_pad_geometry_list(pad_shape=(int(512/bin), int(1024/bin)), pixel_size=75e-6*bin, distance=distance,
                               tiling_shape=(4, 2), pad_gap=36*75e-6)
gap = 9e-3
pads[0].t_vec += + np.array([1, 0, 0])*gap/2 - np.array([0, 1, 0])*gap/2
pads[1].t_vec += + np.array([1, 0, 0])*gap/2 - np.array([0, 1, 0])*gap/2
pads[2].t_vec += - np.array([1, 0, 0])*gap/2 - np.array([0, 1, 0])*gap/2
pads[3].t_vec += - np.array([1, 0, 0])*gap/2 - np.array([0, 1, 0])*gap/2
pads[4].t_vec += + np.array([1, 0, 0])*gap/2 + np.array([0, 1, 0])*gap/2
pads[5].t_vec += + np.array([1, 0, 0])*gap/2 + np.array([0, 1, 0])*gap/2
pads[6].t_vec += - np.array([1, 0, 0])*gap/2 + np.array([0, 1, 0])*gap/2
pads[7].t_vec += - np.array([1, 0, 0])*gap/2 + np.array([0, 1, 0])*gap/2
q_vecs = [pad.q_vecs(beam=beam) for pad in pads]
q_mags = [pad.q_mags(beam=beam) for pad in pads]
solid_angles = [pad.solid_angles() for pad in pads]
polarization_factors = [pad.polarization_factors(beam=beam) for pad in pads]


# Refractive index of water
cmp = xraylib.CompoundParser('H2O')
dens = water_density
MM = cmp['molarMass']
N = dens/(MM/NA/1000)  # Number density of molecules (SI)
ref_idx = 0
E = photon_energy / (1000 * eV)
for i in range(cmp['nElements']):
    Z = cmp['Elements'][i]
    nZ = cmp['nAtoms'][i]
    mf = cmp['massFractions'][i]
    f = xraylib.FF_Rayl(Z, 0) + xraylib.Fi(Z, E) - 1j*xraylib.Fii(Z, E)
    ref_idx += N * mf * f
ref_idx = 1 - (ref_idx * wavelength**2 * r_e / (2*np.pi))

fdens = (1 - ref_idx)*2*np.pi/wavelength**2
print(fdens)


amps = [fdens*sphere_form_factor(radius=drop_radius, q_mags=qm) for qm in q_mags]





# Refractive index is
#
# n = 1 - (1/2pi) Sum: N * r_e * lam^2 * f
#
# For an overall density rho and mass fraction mf, the fractional density is rho*mf
#
#

intensities = []
for i in range(len(pads)):
    pad = pads[i]
    d = np.abs(amps[i])**2*solid_angles[i]*polarization_factors[i]*fluence
    d = np.random.poisson(d)
    intensities.append(pad.reshape(d))

# intensities = [r_e*2*pad.reshape(np.abs(amp)**2) for amp, pad in zip(amps, pads)]

# intensities = [np.random.poisson(a) for a in intensities]

dat = intensities
# dat = [pad.reshape(qmag) for qmag, pad in zip(q_mags, pads)]
print(dat[0].shape)
padview = PADView(raw_data=dat, pad_geometry=pads)
# padview.set_levels(-0.2, 2)
padview.start()
print('done')
