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
import scipy
from reborn.misc.rotate import Rotate3D, Rotate3Dvkfft, have_gpu
from scipy.stats import special_ortho_group
from reborn.math import kabsch  # Legacy
import reborn.math.kabsch  # Legacy
from reborn.misc.rotate import kabsch

def makegaussians(w, g, s, n):
    x = np.linspace(-0.5 * (1.0 - 1.0 / n), 0.5 * (1.0 - 1.0 / n), num=n)
    d = np.zeros((n, n, n), dtype=np.float64)
    for ig in range(len(w)):
        xgauss = np.exp(-0.5 * ((x - g[ig, 0]) / s) ** 2)
        ygauss = np.exp(-0.5 * ((x - g[ig, 1]) / s) ** 2)
        zgauss = np.exp(-0.5 * ((x - g[ig, 2]) / s) ** 2)
        d += w[ig] * np.tile(np.reshape(xgauss, (n, 1, 1)), (1, n, n)) * \
             np.tile(np.reshape(ygauss, (1, n, 1)), (n, 1, n)) * \
             np.tile(np.reshape(zgauss, (1, 1, n)), (n, n, 1))
    return d


rng = np.random.default_rng(1717171717)
Nr = 10
Rs = scipy.spatial.transform.Rotation.random(Nr, random_state=rng)
Ngr = 8
Ngi = 8
sigma = 0.05
gr0 = (rng.random((Ngr, 3)) - 0.5) * sigma * 3.0
wr = rng.random(Ngr) - 0.5
gi0 = (rng.random((Ngi, 3)) - 0.5) * sigma * 3.0
wi = rng.random(Ngi) - 0.5
N = 32
methods = [Rotate3D]  # , rotate3Dl, rotate3Dj]
# if have_gpu:
#     methods.append(Rotate3Dvkfft)
types = [np.complex128, np.complex64, np.float64, np.float32]


def test_01():
    gr = gr0.copy()
    gi = gi0.copy()
    datar = makegaussians(wr, gr, sigma, N)
    data = datar + 1j * makegaussians(wi, gi, sigma, N)
    for t in types:
        for m in methods:
            gr = gr0.copy()
            gi = gi0.copy()
            if t == np.complex128 or t == np.complex64:
                r3df = m(data.astype(t))
                d_max = np.max(np.abs(data))
            else:
                r3df = m(datar.astype(t))
                d_max = np.max(np.abs(datar))
            for ir in Rs:
                gr = ir.apply(gr)
                gi = ir.apply(gi)
                r3df.rotation(ir)
                if t == np.complex128 or t == np.complex64:
                    error = np.max(np.abs(r3df.f - makegaussians(wr, gr, sigma, N)
                                          - 1j * makegaussians(wi, gi, sigma, N))) / d_max
                    assert (error < 1e-4)  # Weak tests; only checks if something is horribly wrong
                else:
                    error = np.max(np.abs(r3df.f - makegaussians(wr, gr, sigma, N))) / d_max
                    assert (error < 1e-4)  # Weak test


def test_kabsch():
    r"""
    Generate a random (3x3) array and rotate it with a random rotation matrix.
    See if we can deduce the rotational matrix given the original A and the rotated A
    using the Kabsch algorithm.
    """

    np.random.seed(42)

    small = 1.0e-12 # Error we want the result to be smaller than

    # Generate a random A matrix
    A = np.random.rand(3,3)

    # Generate a random rotation matrix
    R = special_ortho_group.rvs(dim=3)

    # Rotate A
    A_home = np.dot(R,A)

    # Take the transpose because the scipy align_vectors function assumes shape of (N,3),
    # i.e. the vectors are assumed to be stacked horizontally.
    A = A.T
    A_home = A_home.T

    # Now run the Kabsch algorithm and convert the output to a rotation matrix
    R_est = kabsch(A, A_home)

    assert (np.sum(np.abs(R - R_est)) < small)

