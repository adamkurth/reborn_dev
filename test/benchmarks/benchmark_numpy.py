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
from time import time

R = np.array([[0, 1., 0], [-1, 0, 0], [0, 0, 1.]])
vec = np.array([1, 2, 3])
vec_rotated = np.dot(vec, R.T)
print(R)
print(vec)
print(vec_rotated)

# Rotating vectors
trials = 100
for n in [1, 1000, 1000000]:
    vecs = np.random.rand(n, 3)
    rot = np.random.rand(3, 3)
    t = time()
    for i in range(trials):
        junk1 = np.dot(rot, vecs.T).T
    t1 = time() - t
    t = time()
    for i in range(trials):
        junk2 = np.dot(vecs, rot.T)  # This is most often the winner
    t2 = time() - t
    t = time()
    for i in range(trials):
        junk3 = np.dot(vecs, rot.T.copy())
    t3 = time() - t
    print('%7d :  %10.2e,  %10.2e,  %10.2e' % (n, t1/n/trials, t2/n/trials, t3/n/trials))
