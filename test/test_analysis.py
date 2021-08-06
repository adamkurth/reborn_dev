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
