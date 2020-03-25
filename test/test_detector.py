

import sys

sys.path.append('..')
import reborn as ba
from reborn import detector
from reborn import source
import numpy as np


def make_pad_list():

    pad_geom = []
    pad = ba.detector.PADGeometry()
    pad.t_vec = [0, .01, .5]
    pad.fs_vec = [100e-6, 0, 0]
    pad.ss_vec = [0, 100e-6, 0]
    pad.n_fs = 100
    pad.n_ss = 150
    pad_geom.append(pad)
    pad = ba.detector.PADGeometry()
    pad.t_vec = [0, -.01, .5]
    pad.fs_vec = [100e-6, 0, 0]
    pad.ss_vec = [0, 100e-6, 0]
    pad.n_fs = 100
    pad.n_ss = 150
    pad_geom.append(pad)

    return pad_geom


def test_PADGeometry():

    # Check that the two solid angle calculations are in agreement

    pad = detector.PADGeometry()
    assert pad.t_vec is None
    pad = detector.PADGeometry(pixel_size=100e-6, distance=1, shape=(100, 100))
    assert np.max(pad.solid_angles1() - pad.solid_angles2())/np.max(pad.solid_angles2()) < 1e-6


def test_beam():

    beam = source.Beam()
    beam.wavelength = 1.5e-10
    beam.beam_vec = np.array([0, 0, 1])

    pad_geom = detector.PADGeometry(pixel_size=100e-6, distance=1, shape=(100, 100))
    # pad_geom.simple_setup()

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

    assembler = ba.detector.PADAssembler(pad_geom)
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
    profile = rad.get_profile(dat, average=False)  # Sums over

    print(q_mags)
    print(rad.bins)
    print(rad.bin_edges)

    assert profile[0] == 1
    assert profile[1] == 4
    assert profile[2] == 4

    profile = rad.get_profile(dat, average=True)  # Sums over

    assert profile[0] == 1
    assert profile[1] == 1
    assert profile[2] == 1

    mask = np.ones([3, 3])
    mask[0, 0] = 0

    rad.set_mask(mask)
    profile = rad.get_profile(dat, average=False)  # Sums over
    print(mask)
    print(profile)

    assert profile[0] == 1
    assert profile[1] == 4
    assert profile[2] == 3


def test_radial_profiler_02():

    pad_geom = make_pad_list()

    beam = [0, 0, 1]
    wav = 1.5e-10

    q_mags = np.ravel([p.q_mags(beam_vec=beam, wavelength=wav) for p in pad_geom])

    rad = detector.RadialProfiler()
    rad.make_plan(q_mags, mask=None, n_bins=100, q_range=None)

    data = np.ravel([np.random.rand(p.n_ss, p.n_fs) for p in pad_geom])

    prof = rad.get_profile(data, average=True)
    assert(np.max(prof) <= 1)
    assert(np.min(prof) >= 0)

