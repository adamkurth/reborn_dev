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

import os
import tempfile
from reborn import detector
from reborn import source
from reborn.external import crystfel
import numpy as np
import scipy.constants as const
eV = const.value('electron volt')
np.random.seed(0)
tempdir = tempfile.gettempdir()


def make_pad_list():
    r""" Simply check the creation of a pad list. """
    pad_geom = []
    pad = detector.PADGeometry()
    pad.t_vec = [0, .01, .5]
    pad.fs_vec = [100e-6, 0, 0]
    pad.ss_vec = [0, 100e-6, 0]
    pad.n_fs = 100
    pad.n_ss = 150
    pad_geom.append(pad)
    pad = detector.PADGeometry()
    pad.t_vec = [0, -.01, .5]
    pad.fs_vec = [100e-6, 0, 0]
    pad.ss_vec = [0, 100e-6, 0]
    pad.n_fs = 100
    pad.n_ss = 150
    pad_geom.append(pad)
    return pad_geom


def test_solid_angles():
    # Check that the two solid angle calculations are in agreement
    pad = detector.PADGeometry()
    assert pad.t_vec is None
    pad = detector.PADGeometry(pixel_size=100e-6, distance=1, shape=(100, 100))
    assert np.max(pad.solid_angles1() - pad.solid_angles2())/np.max(pad.solid_angles2()) < 1e-6


def test_save_pad():
    file_name = os.path.join(tempdir, 'test.json')
    pad1 = detector.PADGeometry(pixel_size=100e-6, distance=1, shape=(100, 100))
    pad1.save_json(file_name)
    pad2 = detector.PADGeometry()
    pad2.load_json(file_name)
    os.remove(file_name)
    assert pad1 == pad2


def test_save_pad_list():
    file_name = os.path.join(tempdir, 'test.json')
    pads1 = make_pad_list()
    detector.save_pad_geometry_list(file_name, pads1)
    pads2 = detector.load_pad_geometry_list(file_name)
    # print(pads2)
    for i in range(len(pads1)):
        assert pads1[i] == pads2[i]
    os.remove(file_name)


def test_beam():
    beam = source.Beam()
    beam.wavelength = 1.5e-10
    beam.beam_vec = np.array([0, 0, 1])
    pad_geom = detector.PADGeometry(pixel_size=100e-6, distance=1, shape=(100, 100))
    # TODO: put some thought into these tests...
    out = pad_geom.scattering_angles(beam=beam)
    assert np.min(out >= 0)
    out = pad_geom.polarization_factors(beam=beam)
    assert np.min(out > 0)
    out = pad_geom.ds_vecs(beam=beam)
    assert np.min(np.abs(out) >= 0)
    out = pad_geom.q_mags(beam=beam)
    assert np.min(out >= 0)
    out = pad_geom.q_vecs(beam=beam)
    assert np.min(np.abs(out) >= 0)


def test_PADAssembler():
    # TODO: check that layout is actually correct.
    pad_geom = make_pad_list()
    assembler = detector.PADAssembler(pad_geom)
    dat = [p.ones() for p in pad_geom]
    ass = assembler.assemble_data(dat)
    assert(np.min(ass) == 0)
    assert(np.max(ass) == 1)


def test_radial_profiler_01():
    pad_geom = detector.PADGeometry(shape=[3, 3], distance=1.0, pixel_size=100e-6)
    q_mags = pad_geom.q_mags(beam_vec=[0, 0, 1], wavelength=1.0e-10)
    # q_mags = [8885765.80967349 6283185.28361764 8885765.80967349 6283185.28361764 0. 6283185.28361764
    #           8885765.80967349 6283185.28361764 8885765.80967349]
    dat = np.ones([3, 3])
    rad = detector.RadialProfiler(q_mags=q_mags, mask=None, n_bins=3, q_range=[0, 9283185])
    # rad.bin_edges = [-2320796.25  2320796.25  6962388.75 11603981.25]
    profile = rad.get_sum_profile(dat)  # Sums over
    assert profile[0] == 1
    assert profile[1] == 4
    assert profile[2] == 4
    profile = rad.get_mean_profile(dat)  # Sums over
    assert profile[0] == 1
    assert profile[1] == 1
    assert profile[2] == 1
    mask = np.ones([3, 3])
    mask[0, 0] = 0
    profile = rad.get_sum_profile(dat, mask=mask)  # Sums over
    assert profile[0] == 1
    assert profile[1] == 4
    assert profile[2] == 3
    mask = np.ones([3, 3])
    mask[0, 0] = 0
    rad.set_mask(mask)
    profile = rad.get_sum_profile(dat)  # Sums over
    assert profile[0] == 1
    assert profile[1] == 4
    assert profile[2] == 3
    mask = np.ones([3, 3])
    mask[0, 0] = 0
    rad.set_mask(mask)
    profile = rad.get_mean_profile(dat)  # Sums over
    assert profile[0] == 1
    assert profile[1] == 1
    assert profile[2] == 1


def test_radial_profiler_02():
    pad_geom = make_pad_list()
    beam = [0, 0, 1]
    wav = 1.5e-10
    q_mags = np.ravel([p.q_mags(beam_vec=beam, wavelength=wav) for p in pad_geom])
    rad = detector.RadialProfiler(q_mags=q_mags, mask=None, n_bins=100, q_range=None)
    data = np.ravel([np.random.rand(p.n_ss, p.n_fs) for p in pad_geom])
    prof = rad.get_mean_profile(data)
    assert(np.max(prof) <= 1)
    assert(np.min(prof) >= 0)


def test_radial_profiler_03():
    pads = detector.cspad_2x2_pad_geometry_list()
    beam = source.Beam(wavelength=1.0e-10)
    rad = detector.RadialProfiler(beam=beam, pad_geometry=pads, mask=None, n_bins=100)
    data = pads.random()
    prof1 = rad.get_mean_profile(data)
    prof2 = rad.get_profile_statistic(data, statistic=np.mean)
    prof3, _ = detector.get_radial_profile(data, beam, pads, n_bins=100, statistic=np.mean)
    assert(np.max(np.abs(prof1-prof2)) == 0)
    assert(np.max(np.abs(prof3-prof2)) == 0)


def test_vector_math():

    pad = detector.PADGeometry(shape=(2, 2), distance=0.1, pixel_size=1e-3)
    vecs = pad.position_vecs()
    j, i = pad.vectors_to_indices(vecs, insist_in_pad=False)
    jj, ii = np.indices(pad.shape())
    assert(np.max(np.abs(j - jj.ravel())) < 1e-6)
    assert (np.max(np.abs(i - ii.ravel())) < 1e-6)
    j, i = pad.vectors_to_indices(vecs, insist_in_pad=True, round=True)
    assert(np.max(np.abs(j - jj.ravel())) == 0)
    assert (np.max(np.abs(i - ii.ravel())) == 0)


def test_saving():

    shapes = [np.array((100, 101)) for _ in range(8)]
    masks = [np.round(np.random.random(shapes[i]) * 0.6).astype(int) for i in range(8)]
    assert np.sum(masks[0]) > 0
    file_name_1 = os.path.join(tempdir, 'unpacked.mask')
    file_name_2 = os.path.join(tempdir, 'packed.mask')
    detector.save_pad_masks(file_name_1, masks, packbits=False)
    detector.save_pad_masks(file_name_2, masks)
    unpacked = detector.load_pad_masks(file_name_1)
    # print('loaded unpacked')
    packed = detector.load_pad_masks(file_name_2)
    for i in range(len(masks)):
        assert np.max(packed[i] - unpacked[i]) == 0
        assert np.max(masks[i] - unpacked[i]) == 0
    os.remove(file_name_1)
    os.remove(file_name_2)


def test_standard_pads():
    cspad = detector.cspad_pad_geometry_list()
    cspad2x2 = detector.cspad_2x2_pad_geometry_list()
    pnccd = detector.pnccd_pad_geometry_list()
    epix = detector.epix10k_pad_geometry_list()
    jungfrau = detector.jungfrau4m_pad_geometry_list()
    mpccd = detector.mpccd_pad_geometry_list()
    assert(isinstance(cspad, detector.PADGeometryList))
    assert(isinstance(cspad2x2, detector.PADGeometryList))
    assert (isinstance(pnccd, detector.PADGeometryList))
    assert (isinstance(epix, detector.PADGeometryList))
    assert (isinstance(jungfrau, detector.PADGeometryList))
    assert (isinstance(mpccd, detector.PADGeometryList))


def test_padlist():
    beam = source.Beam(photon_energy=9000*1.602e-19)
    pads = detector.cspad_pad_geometry_list()
    pads2 = detector.epix10k_pad_geometry_list()
    padlist = detector.PADGeometryList(pads)
    padlist2 = detector.PADGeometryList(pads2)
    padlist3 = padlist.copy()
    file_name = os.path.join(tempdir, 'test.json')
    padlist3.save_json(file_name)
    padlist4 = detector.load_pad_geometry_list(file_name)
    padlist5 = detector.PADGeometryList(padlist)
    assert(padlist != padlist2)
    assert(padlist.hash != padlist2.hash)
    assert(padlist.hash == padlist3.hash)
    assert(len(padlist) == 64)
    assert(padlist3 == padlist)
    assert(padlist3 == padlist4)
    assert(padlist.validate() is True)
    assert(padlist.position_vecs().shape[1] == 3)
    assert(padlist.s_vecs().shape[1] == 3)
    assert(padlist.ds_vecs(beam).shape[1] == 3)
    assert(padlist.q_vecs(beam).shape[1] == 3)
    assert(padlist.q_mags(beam).size == padlist.n_pixels)
    assert(padlist.solid_angles().size == padlist.n_pixels)
    assert(padlist.polarization_factors(beam).size == padlist.n_pixels)
    assert(padlist.random().size == padlist.n_pixels)
    assert(isinstance(padlist5, list))
    assert(isinstance(padlist, detector.PADGeometryList))

def test_loading():
    crystfel_geom = crystfel.epix10k_geom_file
    json_geom = detector.epix10k_geom_file
    pads = detector.PADGeometryList()
    pads.load(json_geom)
    assert(len(pads) != 1)
    pads = detector.PADGeometryList()
    pads.load(crystfel_geom)
    assert (len(pads) != 1)
    pads2 = detector.PADGeometryList(filepath=crystfel_geom)
    assert(pads == pads2)
