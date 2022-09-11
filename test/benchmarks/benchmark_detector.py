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
pad = reborn.detector.PADGeometry(pixel_size=100e-6, distance=0.05, shape=(1000, 1000))

n_iter = 10

t0 = time()
for i in range(n_iter):
    a = pad.solid_angles1()
t1 = (time()-t0)/n_iter
print(f'solid_angles1: {t1} milliseconds')

t0 = time()
for i in range(n_iter):
    a = pad.solid_angles2()
t2 = (time()-t0)/n_iter
print(f'solid_angles2: {t2}')
print(f'speedup: {t2/t1}')


# Check if cache helps speed things up
beam = reborn.source.Beam()
geom = reborn.detector.cspad_pad_geometry_list()
geom_cached = reborn.detector.cspad_2x2_pad_geometry_list()
geom_cached.do_cache = True

n_iter = 10

t0 = time()
for i in range(n_iter):
    q = geom.q_vecs(beam=beam)
ta = (time() - t0)*1000/n_iter
print(f"q_vecs, no cache: {ta} milliseconds.")
t0 = time()
for i in range(n_iter):
    q = geom_cached.q_vecs(beam=beam)
tb = (time() - t0)*1000/n_iter
print(f"q_vecs, cached: {tb} milliseconds.")
print(f"speedup: {ta/tb}")
