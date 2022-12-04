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

import time
import numpy as np
# np.set_printoptions(precision=4)
from reborn.fortran import peaks_f
from reborn import analysis
from scipy.signal import convolve


def test_fortran():

    # try:
    #     from reborn.analysis import peaks_f
    # except ImportError:
    #     return

    npx = 1000
    npy = 2000
    x, _ = np.meshgrid(np.arange(0, npx), np.arange(0, npy), indexing='ij')
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
    noise = np.sqrt(((0+1+16)*3-1)/8. - (((0+1+4)*3-1)/8.)**2)
    sig = 1 - ((0+1+4)*3-1)/8.
    assert np.abs(out[1, 1] - sig/noise) < 1e-6

    dat, _ = np.meshgrid(np.arange(0, npx), np.arange(0, npy), indexing='ij')
    peaks_f.boxconv(dat, out, 1)
    assert np.abs(out[1, 1]/9 - dat[1, 1]) == 0


def snr_filter(data, mask, nin, ncent, nout):
    kin = np.ones((2*nin+1, 2*nin+1))
    kout = np.ones((2*nout+1, 2*nout+1))
    d = (nout-ncent)
    kout[d:-d, d:-d] = 0
    dm = data*mask
    d2m = data**2*mask
    cin = convolve(mask, kin, mode='same')
    cout = convolve(mask, kout, mode='same')
    bak = convolve(dm, kout, mode='same') / cout
    sig = convolve(dm, kin, mode='same') / cin - bak
    bak2 = convolve(d2m, kout, mode='same') / cout
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
    t = time.time()
    snr1, sig1 = analysis.masking.snr_filter_test(data, mask, mask, 0, 0, 2)
    print(time.time()-t)
    print('data')
    print(data)
    print('mask')
    print(mask)
    print('snr1')
    print(snr1)
    print('sig1')
    print(sig1)
    assert(sig1[2, 2] == 1 - 16/24)
    assert(sig1[4, 4] == 0)
    data, mask = makedata()
    t = time.time()
    snr2, sig2 = analysis.masking.snr_filter(data, mask, mask, 0, 0, 2)
    print(time.time()-t)
    assert (np.max(np.abs(snr1 - snr2)) < 1e-6)


def notest_snr_02():
    shape = (5, 5)
    data = np.zeros(shape)
    data.flat[0::2] = 1
    data[2, 3] = 4
    # print('')
    # print(data)
    mask = np.ones(shape)
    smask, snr = analysis.masking.snr_mask(data, mask, 0, 0, 2, threshold=6, mask_negative=True, max_iterations=1,
                                     subtract_median=False)
    print('snr')
    print(snr)
    # print(smask)
    # print(snr)
    # meen = (np.sum(data) - data[2, 2])/(np.sum(mask) - mask[2, 2])
    # dif = data - meen
    # var = (np.sum(dif**2) - dif[2, 2]**2)/(np.sum(mask) - mask[2, 2])
    # sdev = np.sqrt(var)
    # snr1 = dif[2, 2] / sdev
    # print(snr[2, 2], snr1)


    snr3, sig3 = analysis.masking.snr_filter(data, mask, mask, 0, 0, 2)
    snr4, sig4 = analysis.masking.boxsnr(data, mask, mask, 0, 0, 2)
    print('snr3')
    print(snr3)
    # assert(np.max(np.abs(snr-snr3)) < 1e-6)
    assert(np.max(np.abs(snr3-snr4)) < 1e-6)
    # print(snr3)
    # broke
    # padding = ((2, 2), (2, 2))
    # data = np.pad(data, padding)
    # mask = np.pad(mask, padding)
    #
    # data *= mask
    # sk = np.zeros(shape)  # Inner signal kernel
    # sk[2, 2] = 1
    # ak = np.ones(shape)  # Outer annulus kernel
    # ak[2, 2] = 0
    # sk = np.pad(sk, padding)
    # ak = np.pad(ak, padding)
    # sk = np.roll(sk, (-2, -2), axis=(0, 1))
    # ak = np.roll(ak, (-2, -2), axis=(0, 1))
    # print('data')
    # print(data)
    # print('mask')
    # print(mask)
    # print('sk')
    # print(sk)
    # print('ak')
    # print(ak)
    # mode = 'same'
    # sc = convolve(mask, sk, mode=mode)[0:-4, 0:-4]  # Signal pixel count
    # print('sc')
    # print(sc)
    # ac = convolve(mask, ak, mode=mode)[0:-4, 0:-4]  # Annulus pixel count
    # print('ac')
    # print(ac)
    # ba = convolve(data, ak, mode=mode)[0:-4, 0:-4] / ac  # Average background
    # print('ba')
    # print(ba)
    # b2a = convolve(data**2, ak, mode=mode)[0:-4, 0:-4] / ac  # Average background squared
    # sa = convolve(data, sk, mode=mode)[0:-4, 0:-4] / sc  # Integrated signal
    # da = sa - ba  # Average difference
    # sig = np.sqrt(b2a - ba**2)
    # std = da / sig / np.sqrt(sc)
    # print('std')
    # print(std)
