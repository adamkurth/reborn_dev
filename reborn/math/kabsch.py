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

import scipy.spatial.transform


def kabsch(A, A0):
    r"""
    Finds the rotation matrix that will bring a set of vectors A into alignment with 
    another set of vectors A0. Uses the Kabsch algorithm implemented in 
    scipy.spatial.transform.Rotation.align_vectors

    Arguments:
        A (|ndarray|): N, 3x1 vectors stacked into the shape (N,3) 
        A0 (|ndarray|): N, 3x1 vectors stacked into the shape (N,3) 

    Returns:
        |ndarray|: 3x3 rotation matrix.
    """
    return scipy.spatial.transform.Rotation.align_vectors(A0, A)[0].as_matrix()
