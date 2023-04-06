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
from reborn.fortran import peaks_f
from reborn import analysis
from scipy.signal import convolve
from reborn.simulate.form_factors import sphere_form_factor


def test_fortran():
    npx = 1000
    npy = 2000
    x, _ = np.meshgrid(np.arange(0, npx), np.arange(0, npy), indexing="ij")
    assert x.shape[0] == npx
    dat = x**2
    dat = dat.astype(np.float64)
    dat = np.asfortranarray(dat)
    assert dat[1, 1] == 1
    mask = np.ones((npx, npy))
    mask = np.asfortranarray(mask)
    out = np.empty_like(dat)
    out = np.asfortranarray(out)
    signal = np.asfortranarray(out)
    nin = 0
    ncent = 0
    nout = 1
    peaks_f.boxsnr(dat, mask, mask, out, signal, nin, ncent, nout)
    noise = np.sqrt(((0 + 1 + 16) * 3 - 1) / 8.0 - (((0 + 1 + 4) * 3 - 1) / 8.0) ** 2)
    sig = 1 - ((0 + 1 + 4) * 3 - 1) / 8.0
    assert np.abs(out[1, 1] - sig / noise) < 1e-6

    dat, _ = np.meshgrid(np.arange(0, npx), np.arange(0, npy), indexing="ij")
    peaks_f.boxconv(dat, out, 1)
    assert np.abs(out[1, 1] / 9 - dat[1, 1]) == 0


def snr_filter(data, mask, nin, ncent, nout):
    kin = np.ones((2 * nin + 1, 2 * nin + 1))
    kout = np.ones((2 * nout + 1, 2 * nout + 1))
    d = nout - ncent
    kout[d:-d, d:-d] = 0
    dm = data * mask
    d2m = data**2 * mask
    cin = convolve(mask, kin, mode="same")
    cout = convolve(mask, kout, mode="same")
    bak = convolve(dm, kout, mode="same") / cout
    sig = convolve(dm, kin, mode="same") / cin - bak
    bak2 = convolve(d2m, kout, mode="same") / cout
    stderr = np.sqrt((bak2 - bak**2) / cin)
    snr = sig / stderr
    return snr


def test_snr_01():
    def makedata():
        shape = (5, 5)
        data = np.zeros(shape)
        data.flat[0::2] = 1
        data[2, 3] = 4
        mask = np.ones(shape)
        return data, mask
    data, mask = makedata()
    snr1, sig1 = analysis.masking.snr_filter_test(data, mask, mask, 0, 0, 2)
    assert sig1[2, 2] == 1 - 16 / 24
    assert sig1[4, 4] == 0
    data, mask = makedata()
    snr2, sig2 = analysis.masking.snr_filter(data, mask, mask, 0, 0, 2)
    assert np.max(np.abs(snr1 - snr2)) < 1e-6


def notest_snr_02():
    shape = (5, 5)
    data = np.zeros(shape)
    data.flat[0::2] = 1
    data[2, 3] = 4
    mask = np.ones(shape)
    smask, snr = analysis.masking.snr_mask(
        data,
        mask,
        0,
        0,
        2,
        threshold=6,
        mask_negative=True,
        max_iterations=1,
        subtract_median=False,
    )
    snr3, sig3 = analysis.masking.snr_filter(data, mask, mask, 0, 0, 2)
    snr4, sig4 = analysis.masking.boxsnr(data, mask, mask, 0, 0, 2)
    assert np.max(np.abs(snr3 - snr4)) < 1e-6


def test_fit_sphere_profile():
    q_mags = np.linspace(0, 1e10, 617)
    r = np.array([1, 10, 20, 50]) * 1e-9
    sd = analysis.optimize.SphericalDroplets(q=q_mags, r=r)
    test_sphere_diffraction = (
        sphere_form_factor(radius=10e-9, q_mags=q_mags, check_divide_by_zero=True)
    ) ** 2
    r_min, r_dic = sd.fit_profile(I_D=test_sphere_diffraction)
    A_min = r_dic["A_min"]
    assert r_min == 10e-9
    assert A_min == 1
