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
Water 1D Profiles
========================

Simple example of how to access and plot a water radial profile.

Contributed by Kosta Karpos
"""

import numpy as np
import matplotlib.pyplot as plt
from reborn.simulate.solutions import water_scattering_factor_squared

q_range = [0, 3.2]
temperatures = [273, 280, 320]

# %%
# Setup q_range in SI units (meters)
q_range = np.linspace(0, q_range, 1000) * 1e10

# %%
# Get the mean radial profiles for each temperature
means = [water_scattering_factor_squared(q_range, temperature=t) for t in temperatures]

# %%
# It's plottin' time
for t, r in zip(temperatures, means):
    print(t)
    plt.plot(q_range * 1e-10, r, label=t.__str__())
    break

plt.legend(title="Temperatures")
# plt.grid(alpha=0.5)
plt.title("Water Radial Profiles")
plt.xlabel(r"q = 4$\pi \sin{ \theta} / \lambda$ [$\AA^{-1}$]")
plt.ylabel("I(q)")
plt.show()
