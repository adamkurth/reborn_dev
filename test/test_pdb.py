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
from reborn.target import crystal


def test_nonzero_u_vec():

    # Note that the following is found in the pdb file 1lsp.pdb
    # We have a non-zero U vector.

    # SCALE1      0.010906  0.000000  0.000000       -0.25000
    # SCALE2      0.000000  0.015307  0.000000       -0.25000
    # SCALE3      0.000000  0.000000  0.026212       -0.25000

    # REMARK 290 CRYSTALLOGRAPHIC SYMMETRY
    # REMARK 290 SYMMETRY OPERATORS FOR SPACE GROUP: P 21 21 2
    # REMARK 290
    # REMARK 290      SYMOP   SYMMETRY
    # REMARK 290     NNNMMM   OPERATOR
    # REMARK 290       1555   X,Y,Z
    # REMARK 290       2555   -X,-Y,Z
    # REMARK 290       3555   -X+1/2,Y+1/2,-Z
    # REMARK 290       4555   X+1/2,-Y+1/2,-Z
    # REMARK 290
    # REMARK 290     WHERE NNN -> OPERATOR NUMBER
    # REMARK 290           MMM -> TRANSLATION VECTOR

    W_correct = [np.array([[ 1., 0.,0.],
                           [ 0., 1.,0.],
                           [ 0., 0.,1.]]),
                 np.array([[-1., 0., 0.],
                           [ 0.,-1., 0.],
                           [ 0., 0., 1.]]),
                 np.array([[-1., 0., 0.],
                           [ 0., 1., 0.],
                           [ 0., 0.,-1.]]),
                 np.array([[ 1., 0., 0.],
                           [ 0.,-1., 0.],
                           [ 0., 0.,-1.]])]
    Z_correct = [np.array([0, 0, 0]),
                 np.array([0, 0, 0]),
                 np.array([0.5, 0.5, 0]),
                 np.array([0.5, 0.5, 0])]

    pdb_file = crystal.get_pdb_file('1lsp')

    cryst = crystal.CrystalStructure(pdb_file, no_warnings=True)

    dic = crystal.pdb_to_dict(pdb_file)
    S = dic['scale_matrix']
    Sinv = np.linalg.inv(S)
    U = dic['scale_translation']
    W = []
    Z1 = []
    Z2 = []
    Z3 = []
    for i in range(len(dic['spacegroup_translations'])):
        T = dic['spacegroup_translations'][i]
        R = dic['spacegroup_rotations'][i]
        W.append(np.dot(S, np.dot(R, Sinv)))
        Z1.append(np.dot(S, T) + U) # Either this or the next option should be correct
        Z2.append(np.dot(S, T) + np.dot(np.eye(3)-W, U))
        Z3.append(np.dot(S, T))     # This is supposed to be wrong, but it gives the correct result

    # The following two tests fail:
    # for i in range(4):
    #     assert np.max(np.abs(Z2[i] - Z_correct[i])) < 1e-3
    # for i in range(4):
    #     assert np.max(np.abs(Z1[i] - Z_correct[i])) < 1e-3

    # This one succeeds:
    for i in range(4):
        assert np.max(np.abs(Z3[i] - Z_correct[i])) < 1e-3
        assert np.max(np.abs(W[i] - W_correct[i])) < 1e-3
        assert np.max(np.abs(cryst.spacegroup.sym_translations[i] - Z_correct[i])) < 1e-3
        assert np.max(np.abs(cryst.spacegroup.sym_rotations[i] - W_correct[i])) < 1e-3
