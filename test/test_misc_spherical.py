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
import numpy as np
from scipy.special import sph_harm
from reborn.misc.spherical import ylmIntegration


def test_ylmIntegration():
    small = 1.0e-12
    L = 15
    f = np.zeros([2 * L + 2, 2 * L + 2], dtype=np.complex128)
    intylm = ylmIntegration(L)
    for l in range(L + 1):
        for m in range(-l, l + 1):
            for j in range(2 * L + 2):
                f[j, :] = sph_harm(m, l, intylm.phi[:], intylm.theta[j])
            ylmc = intylm.calc_ylmcoef(f)
            ylmc[l, m] -= 1.0
            assert (np.max(np.abs(ylmc)) < small)
