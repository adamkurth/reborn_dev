# This file is part of reborn <https://kirianlab.gitlab.io/reborn/>.
#
# reborn is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# reborn is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with reborn.  If not, see <https://www.gnu.org/licenses/>.

r"""
Water droplet in vacuum
=======================

Diffraction from a water droplet in vacuum.

Contributed by Richard Kirian.

Imports:
"""

import numpy as np
from reborn.source import Beam
from reborn.target import atoms
from reborn.simulate.form_factors import sphere_form_factor
from reborn.detector import jungfrau4m_pad_geometry_list
from reborn.viewers.qtviews import view_pad_data
import scipy.constants as const

# %%
# Define some constants and other parameters that we'll need:

np.random.seed(0)  # Make random numbers that are reproducible
eV = const.value('electron volt')
r_e = const.value('classical electron radius')
NA = const.value('Avogadro constant')
h = const.h
c = const.c
water_density = 1000  # SI units, like everything else in reborn!
photon_energy = 2000*eV
detector_distance = .5
pulse_energy = 1e-3
drop_radius = 20e-9
wavelength = h*c/photon_energy

# %%
# Set up the x-ray source:

beam = Beam(photon_energy=photon_energy, diameter_fwhm=1e-6, pulse_energy=pulse_energy)
fluence = beam.photon_number_fluence

# %%
# Construct a |Jungfrau| 4M detector, made up of 8 modules arranged around a 9mm beam hole.  The number of pixels per
# module is 1024 x 512 and the pixel size is 75 microns.

pads = jungfrau4m_pad_geometry_list(detector_distance=detector_distance)
pads = pads.binned(3)

# %%
# Below we use |xraylib| to get the refractive index of water.  It is equal to
#
# .. math::
#
#     n(\lambda) = 1 - \frac{r_e \lambda^2 }{2\pi} \sum_n N_n f_n(\lambda)
#
# where :math:`N_n` is the number density of scatterer :math:`n` with scattering factor :math:`f_n(\lambda)`.

ref_idx = atoms.xraylib_refractive_index(compound='H2O', density=1000, beam=beam)

# %%
# If you know the refractive index :math:`n(\vec{r})`, then the scattering intensity under the 1st Born approximation is
#
# .. math::
#
#     I(\vec{q}) = J_0 \Delta\Omega P(\vec{q}) \left| \int f(\vec{r}) \exp(i \vec{q}\cdot\vec{r}) d^3r\right|^2
#
# where the scattering density is :math:`f(\vec{r}) = \frac{2\pi}{\lambda^2}(1- n(\vec{r}))`, :math:`\Delta\Omega` is
# the detector (pixel) solid angle, :math:`P(\vec{q})` is a polarization factor, and :math:`J_0` is the incident x-ray
# fluence (photons/area).  Our sphere has uniform refractive index, so we have
#
# .. math::
#
#     I(\vec{q}) &= J_0 \Delta\Omega P(\vec{q}) |f|^2 \left| \int_\text{sphere}  \exp(i \vec{q}\cdot\vec{r}) d^3r\right|^2 \\
#                &= J_0 \Delta\Omega P(\vec{q}) |f|^2 \left| F_\text{sphere}(q) \right|^2
#
# where :math:`F_\text{sphere}(\vec{q})` is the "form factor" of a sphere (i.e. the Fourier transform of a uniform
# sphere).  We can look up all of these needed quantities in reborn:

fdens = (1 - ref_idx)*2*np.pi/wavelength**2  # Scattering density
intensities = []  # We loop over all detector panels
for i in range(len(pads)):
    q_mags = pads[i].q_mags(beam=beam)
    solid_angles = pads[i].solid_angles()
    polarization_factors = pads[i].polarization_factors(beam=beam)
    amps = fdens*sphere_form_factor(radius=drop_radius, q_mags=q_mags)
    d = np.abs(amps)**2*solid_angles*polarization_factors*fluence
    d = np.random.poisson(d)  # Add some Poisson noise
    intensities.append(pads[i].reshape(d))

# %%
# Here is the final result, viewed on a log scale:

dispim = [np.log10(a + 1) for a in intensities]
view_pad_data(data=dispim, pad_geometry=pads)
