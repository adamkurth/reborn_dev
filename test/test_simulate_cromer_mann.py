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
from reborn.simulate import clcore


# def test_no_atomic_form_factor():
#
#     core = clcore.ClCoreDerek(double_precision=False)
#
#     Npix = 512*512
#     Natom = 1000
#
#     q = np.random.random((Npix, 3))
#     r = np.random.random((Natom, 3))
#     f = np.ones(Natom, dtype=np.complex64)
#     A = core.phase_factor_qrf(q, r, f)
#
#     # print (A[0])
# #   now test the cromermann simulation
# #     print("Testing cromermann")
#     core.init_amps(Npix)
#     core.prime_cromermann_simulator(q, None)
#     q_cm = core.get_q_cromermann()
#     r_cm = core.get_r_cromermann(r, sub_com=False)
#     core.run_cromermann(q_cm, r_cm, rand_rot=False)
#     A2 = core.release_amplitudes()
#
#     assert(np.allclose(A, A2))

#
# if __name__ == "__main__":
#
#     test_no_atomic_form_factor()
