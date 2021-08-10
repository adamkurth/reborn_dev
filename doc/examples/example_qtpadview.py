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
Testing PADView
===============

Testing.

Contributed by Richard Kirian.

"""

from reborn import source, detector, simulate
from reborn.viewers.qtviews import PADView2

# %%
# First create some simulated data to look at

beam = source.Beam(wavelength=1.5e-10)
pads = detector.cspad_pad_geometry_list()
dat = simulate.solutions.get_water_profile(pads.q_mags(beam=beam))
pv = PADView2(raw_data=dat, pad_geometry=pads, beam=beam)
pv.show()

# %%
# Test 2

pads = detector.cspad_pad_geometry_list()
dat = [p.random() for p in pads]
pv = PADView2(raw_data=dat, pad_geometry=pads, beam=beam)
pv.show()

# %%
# Did it work?