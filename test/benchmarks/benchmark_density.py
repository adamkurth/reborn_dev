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

import time
import numpy as np
from reborn.target import density

np.random.seed(0)

shape = np.array([10, 10, 10], dtype=int)
densities = np.random.random(shape).astype(np.float64)
x_min = np.array([0, 0, 0], dtype=np.float64)
x_max = np.array([1, 1, 1], dtype=np.float64)
vecs = np.random.rand(80000, 3)
vals = np.zeros(80000, dtype=np.float64)

t = time.time()
for i in range(100):
    density.trilinear_interpolation(densities, vecs, x_min=x_min, x_max=x_max, out=vals)
print(time.time()-t)
print(vals[0])
