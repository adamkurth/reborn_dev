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

import reborn
from time import time
pad = reborn.detector.PADGeometry(pixel_size=100e-6, distance=0.05, n_pixels=1000)

n_iter = 10

t0 = time()
for i in range(n_iter):
    a = pad.solid_angles1()
t1 = (time()-t0)/n_iter
print('t1 (solid_angles1):', t1)

t0 = time()
for i in range(n_iter):
    a = pad.solid_angles2()
t2 = (time()-t0)/n_iter
print('t2 (solid_angles2):', t2)

print('t2/t1:', t2/t1)
