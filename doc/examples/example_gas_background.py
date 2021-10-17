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
Simulating Gas Background
=========================

Many XFEL experiments are done outside of a vacuum environment, typically in helium or air. Here we combine multiple
tools in reborn to simulate the gas background contribution to a scattering profile.

Contributed by Konstantinos Karpos.

The general idea for this simulation is the following.

1. Define the beam and user parameters
2. Set the detector distance to the 'starting' position
3. Calculate the q_mags for the new detector distance
4. Calculate polarization factors and solid angles for the new detector distance
5. Calculate the scattering factors for the new detector distance
6. Calculate the intensity $I(q)$ for the new detector distance
7. Redefine the detector distance (add some dx)
8. Iterate from step 3.
"""

# %%
# Setting up the Beam and General Simulation Parameters
# -----------------------------------------------------
#
# Let's get all our import statements out the way first.
from scipy import constants as const
import numpy as np
import pylab as plt
import reborn
from reborn.detector import RadialProfiler
from reborn.simulate import gas
from reborn.viewers.mplviews import view_pad_data

# %%
# Now define a config dictionary that holds all the simulation parameters and constants. This format is not exactly
# necessary, but it does help keep all the user-defined parameters in one easy-to-find place.

eV = const.value('electron volt')  # J
kb = const.value('Boltzmann constant')  # J  K^-1
r_e = const.value('classical electron radius')

config = {'photon_energy': 10000 * eV,  # J
          'temperature': 293.15,  # K
          'pressure': 101325.0,  # Pa = 1 atm
          'n_steps': 20,  # number of helium chamber spatial steps
          'bin_pixels': 10,  # pixel bin size
          'detector_distance': 0.9,  # m
          'beamstop_diameter': 0.008}  # m

# %%
# One of the first steps for any XRay diffraction simulation using `reborn` is to define your beam. This is done easily
# with the |Beam| class.

beam = reborn.source.Beam(photon_energy=config['photon_energy'])

# %%
# Next we define our detector with the |PADGeometry| class. For simplicity, we'll use the Rayonix MX340 detector. Since
# this detector is rather large, this example also shows how to use the `PADGeometry.binned()` method to bin the pixels.

pad = reborn.detector.rayonix_mx340_xfel_pad_geometry_list(detector_distance=config['detector_distance'])
pad = pad.binned(10)

# %%
# For now, we will consider the gas propagating through the entire detector distance. In reality, there exists some path
# length of gas that is different from the overall detector distance. We will show how to handle this later in the
# example.

# %%
# Finally, we need to define the list of spatial steps to iterate across.

iter_list = np.linspace(1e-5, config['detector_distance'], config['n_steps'])
dx = iter_list[1] - iter_list[0]

# %%
# The number `1e-5` simply is there to avoid errors at 0. 

# %%
# Calculate the Number of Molecules
# ---------------------------------

# %%
# We want to know the number of molecules in the path of the XFEL at each iteration. We assume the gases behave as an
# ideal gas (air and helium). Second, the gas fills the 'chamber' so that number of molecules at each iteration is the
# same.


#
# .. figure:: figures/nmolecules_xfel.svg
#     :scale: 80 %
#     :alt: n molecules
#
#     Diagram for the number of molecules in the path of the XFEL.

# %%
# For an ideal gas, we have the equation 
#
# .. math::
#
#     PV = n k_B T

# %%
# Here, the pressure and temperature are user defined, $k_B$ is the Boltzmann constant, $n$ is the number of molecules,
# and $V$ is the volume. Regarding the volume, the figure above shows the geometry of the problem. We essentially want
# to know the number of molecules within a the path of the beam, which means we must calculate the number of molecules
# within a cylinder defined by two parameters; the beam diameter and the step size of the simulation. In the code block
# below, we account for these parameters and solve for $n$ in the ideal gas equation.

volume = np.pi * dx * (beam.diameter_fwhm/2) ** 2  # volume of a cylinder
n_molecules = config['pressure'] * volume / (kb * config['temperature'])

# %%
# You can use any gas you'd like by utilizing the `reborn.simulate.gas` class, but for the sake of this example, we will
# use helium. This choice isn't arbitrary, in typical XFEL experiments, the beam usually travels through air (72% N2,
# 18% O2), helium, or is in vacuum. It is rare that other substances are used, primarily due to the totally scattered
# intensity increasing with the number of electrons per atom (molecule).

# %%
# Run the Simulation Loop
# -----------------------

# %%
# Now it's time to actually run the simulation. First, initialize your final intensity array. This will dramatically
# increase the speed at which this runs.

I_total = pad.zeros()

# make a copy of the pad so you don't mess with the original
_pad = pad.copy()

alpha = r_e ** 2 * beam.photon_number_fluence
for step in iter_list:
    for p in _pad:  # change the detector distance
        p.t_vec[2] = step
    q_mags_0 = _pad.q_mags(beam=beam)  # calculate the q_mags for that particlar distance
    polarization = _pad.polarization_factors(beam=beam)  # calculate the polarization factors
    solid_angles = _pad.solid_angles2()  # Approximate solid angles
    scatt = gas.isotropic_gas_intensity_profile(molecule='He', beam=beam, q_mags=q_mags_0)  # 1D intensity profile
    F2 = np.abs(scatt) ** 2 * n_molecules 
    I = alpha * polarization * solid_angles * F2  # calculate the scattering intensity
    I = np.random.poisson(I).astype(np.double)  # add in some Poisson noise for funsies
    I_total += I  # sum the results

# %%
# Cool, so now let's plot the final results.

view_pad_data(pad_data=I_total, pad_geometry=pad, vmin=0, vmax=2e4)

# %%
# Radials
# -------

# %%
# For the fun of it, let's add a mask for the forward scatter, view it, then calculate the radial profile of this
# pattern.

mask = pad.ones()
theta = np.arctan(config['beamstop_diameter']/config['detector_distance'])
mask[pad.scattering_angles(beam) < theta] = 0

# %% 
# Let's view the mask quickly. Note that you normally don't want to multiply by the mask, as seen below. You want to
# simply ignore those index values.  For now, the `view_pad_data()` function does not have a way to work with the masks,
# future versions will include this.

view_pad_data(pad_data=I_total * mask, pad_geometry=pad, vmin=0, vmax=2e4)

# define the q_range
q_mags = pad.q_mags(beam=beam)
q_range = np.linspace(np.min(q_mags), np.max(q_mags), 1000)

# Make a RadialProfiler class instance
profiler = RadialProfiler(beam=beam, pad_geometry=pad, n_bins=1000, q_range=[q_range[0], q_range[-1]], mask=mask)

# calculate the mean radial profile
mean_radial = profiler.get_mean_profile(I_total)

plt.figure(figsize=(12, 5))
plt.plot(np.array(q_range)*1e-10, mean_radial)
plt.xlabel(r'q = 4$\pi$ $\sin$($\theta$) / $\lambda$   [$\AA^{-1}$]')
plt.ylabel(r'I(q)  [photons/pixel]')
plt.show()

# %% 
# You may be wondering what's happening at the high q values. This is due to pixel binning. If we were to not do any
# binning, the code take 10x longer to run, but the radial profile would be much smoother.  We do not want lengthy
# calculations our examples because all code is run on the free gitlab servers every time we update reborn.
