import sys

import numpy as np

sys.path.append('..')
import bornagain as ba


def test_all():
    """
    Incomplete test; only check for crashes...
    """

    pad = ba.detector.PADGeometry()
    wavelength = 1.5e-10

    pad.simple_setup(n_pixels=100, pixel_size=1000e-6, distance=0.1)
    q_mags = ba.utils.vec_mag(pad.q_vecs(wavelength=wavelength, beam_vec=[0, 0, 1]))
    dat = np.ones(pad.shape())
    mask = np.ones(pad.shape())

    rp = ba.scatter.RadialProfile()
    n_qbins = 100
    q_range = np.array([0.1, 3]) * 1e10
    rp.make_plan(q_mags=q_mags, mask=mask, n_bins=n_qbins, q_range=q_range)

    profile = rp.get_profile(dat)

    assert (np.max(profile) == 1)
    assert (np.min(profile) == 0)
