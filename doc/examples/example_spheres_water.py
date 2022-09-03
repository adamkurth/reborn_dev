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
Simulation of spheres in water
==============================

Simple simulation of diffraction from a liquid sheet containing spherical scatterers.

Contributed by Richard Kirian.

We start by defining all the relevant parameters.  As always, everything in reborn is in SI units.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import reborn
from reborn.simulate import solutions, form_factors
from reborn.viewers.qtviews import PADView
from reborn.detector import RadialProfiler
from scipy import constants as const

r_e = const.value('classical electron radius')
eV = const.value('electron volt')

detector_shape = [2000, 2000]
pixel_size = 75e-6
detector_distance = 0.1  # Sample to detector distance
water_thickness = 100e-6  # Assuming a sheet of liquid of this thickness
n_shots = 1000  # Number of shots to integrate
n_photons = 1e7  # Photons per shot
photon_energy = 8000*eV  # Photon energy
beam_divergence = 2e-3  # Beam divergence (assuming this limits small-q)
beam_diameter = 5e-6  # X-ray beam diameter (doesn't really matter for solutions scattering)
protein_radius = 10e-9  # Radius of our spherical "protein molecule"
protein_density = 1.34 * 1e3  # Density of spherical protein (g/cm^3, convert to SI kg/m^3)
protein_concentration = 10  # Concentration of protein (mg/ml, which is same as SI kg/m^3)

# %%
# We make a detector and beam, from which we get quantities such as q-vector magnitudes, pixel solid angles, etc.  The
# diffraction intensity is
#
# .. math::
#
#     I(q) = J r_e^2 \Delta \Omega P(\vec{q}) |F(q)|^2

pad = reborn.detector.PADGeometry(distance=detector_distance, shape=detector_shape, pixel_size=pixel_size)
beam = reborn.source.Beam(photon_energy=photon_energy, diameter_fwhm=beam_diameter,
                          pulse_energy=n_photons*photon_energy)
q = pad.q_vecs(beam=beam)
q_mags = pad.q_mags(beam=beam)
J = beam.photon_number_fluence
P = pad.polarization_factors(beam=beam)
SA = pad.solid_angles()

# %%
# We'll make a simple mask to block out the very low angle scattering when we display the pattern:

mask = pad.beamstop_mask(beam=beam, min_angle=beam_divergence)

# %%
# reborn has tabulated scattering factors for water, which come from the work of Greg Hura et al..  These tabulations
# correspond to individual water molecules -- that is, we have a scattering cross-section :math:`f_w(q)` *per molecule*.
# So we need to count the number of water molecules :math:`N_w` in order to estimate the actual scattering intensity
# :math:`I_w(q) \propto N_w |f_w(q)|^2`.

n_water_molecules = water_thickness * np.pi * (beam.diameter_fwhm / 2) ** 2 * solutions.water_number_density()
F_water = solutions.water_scattering_factor_squared(q_mags)
F2_water = F_water**2*n_water_molecules

# %%
# The diffraction from our spherical "proteins" is proportional to the number of spheres. Moreover, should account for
# the fact that the low-angle scattering profile is due to the electron density *contrast* against the surrounding
# water.  At high scattering angles, the diffraction profile is due to inter-atomic distances, but our perfect-sphere
# model does not have any atomic detail of course.

m_protein = protein_density * 4 * np.pi * protein_radius ** 3 / 3  # Spherical protein mass
n = protein_concentration / m_protein  # Number density of spherical proteins
n_protein_molecules = water_thickness * np.pi * (beam.diameter_fwhm / 2) ** 2 * n

# %%
# The scattering cross-section of a sphere comes from the Fourier transform, and the resulting formula is included in
# reborn's form factors module.

F_sphere = form_factors.sphere_form_factor(radius=protein_radius, q_mags=q_mags)
F_sphere *= (protein_density - 1000)/1000 * 3.346e29  # Protein-water contrast.  Water electron density is 3.35e29.
F2_sphere = n_protein_molecules * np.abs((F_sphere**2))
F2 = F2_water + F2_sphere

# %%
# Finally, we compute the scattering intensity in photon units and we add Poisson noise to the result:

I = n_shots * r_e**2 * J * P * SA * F2
I = np.random.poisson(I).astype(np.double)

# %%
# Note that diffraction patterns are usually not 2D arrays.  This is due to the fact that XFELs usually have segmented
# detectors.  It is therefore best to think of diffraction patterns as a collection of little pixel detectors at
# various scattering angles.  In this particular demonstration, we only have one detector panel, so we will make it into
# a 2D array for subsequent display:

I = pad.reshape(I)

# %%
# Let's make a radial profile of the resulting pattern.  reborn has a class that makes it easy to generate radial
# profiles, given PADGeometry and Beam instances.

profiler = RadialProfiler(pad_geometry=pad, beam=beam, mask=mask, n_bins=500, q_range=(0, np.max(q_mags)))
prof = profiler.get_mean_profile(I)

# %%
# Finally, we display the results

plt.figure()
plt.imshow(np.log10(I+10), cmap='gnuplot2', interpolation='nearest')
plt.colorbar()
plt.title('WAXS from Spheres in Water')

plt.figure()
plt.imshow(np.log10(I+1), cmap='gnuplot2', interpolation='nearest')
plt.colorbar()
plt.title('SAXS from Spheres in Water')
plt.xlim([900, 1100])
plt.ylim([900, 1100])

plt.figure()
plt.semilogy(profiler.bin_centers*1e-10, prof)
plt.title('Scattering Profile')
plt.ylabel(r'$I(q)$ [photons/pixel]')
plt.xlabel(r'$q = 4\pi\sin(\theta/2)/\lambda$ [$\AA{}^{-1}$]')

plt.show()
