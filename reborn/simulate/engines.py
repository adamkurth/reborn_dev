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
from scipy.spatial.transform import Rotation
from .clcore import ClCore
from ..fileio.getters import FrameGetter
from ..target.crystal import CrystalStructure, CrystalDensityMap
from ..utils import rotation_about_axis, random_unit_vector, random_beam_vector, max_pair_distance, ensure_list


class MoleculePatternSimulator(object):

    def __init__(self, beam=None, molecule=None, pad=None, oversample=10, max_mesh_size=200, clcore=None):

        if clcore is None:
            clcore = ClCore(group_size=32)
        self.clcore = clcore
        self.pad = pad
        self.molecule = molecule
        self.beam = beam
        self.oversample = oversample
        self.q_vecs = pad.q_vecs(beam=beam)
        self.f = molecule.get_scattering_factors(beam=beam)
        self.intensity_prefactor = pad.f2phot(beam=beam)
        self.resolution = pad.max_resolution(beam=beam)
        self.mol_size = max_pair_distance(molecule.coordinates)
        self.qmax = 2 * np.pi / self.resolution
        self.mesh_size = int(np.ceil(10 * self.mol_size / self.resolution))
        self.mesh_size = int(min(self.mesh_size, max_mesh_size))
        self.a_map_dev = clcore.to_device(shape=(self.mesh_size ** 3,), dtype=clcore.complex_t)
        self.q_dev = clcore.to_device(self.q_vecs, dtype=clcore.real_t)
        self.a_out_dev = clcore.to_device(dtype=clcore.complex_t, shape=self.q_vecs.shape[0])
        clcore.phase_factor_mesh(self.molecule.coordinates, self.f, N=self.mesh_size, q_min=-self.qmax,
                                 q_max=self.qmax, a=self.a_map_dev)

    def generate_pattern(self, rotation=None, poisson=False):

        if rotation is None:
            rotation = Rotation.random().as_matrix()
        self.clcore.mesh_interpolation(self.a_map_dev, self.q_dev, N=self.mesh_size, q_min=-self.qmax,
                                       q_max=self.qmax, R=rotation, a=self.a_out_dev)
        if poisson:
            intensity = np.random.poisson(self.intensity_prefactor * np.abs(self.a_out_dev.get()) ** 2)
        else:
            intensity = self.intensity_prefactor * np.abs(self.a_out_dev.get()) ** 2
        return intensity
