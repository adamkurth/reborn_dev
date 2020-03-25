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
