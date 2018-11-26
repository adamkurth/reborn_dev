import sys
sys.path.append('..')
import numpy as np
from bornagain.analysis import peaks_f


def test_peak_snr_filter():

    nf = 10
    ns = 8
    data = np.random.random((ns, nf)).copy('f')
    mask = np.zeros_like(data, order='f')
    mask[0:2, 0:4]
    snr = np.zeros_like(data, order='f')
    signal = np.zeros_like(data, order='f')
    a = 1
    b = 2
    c = 3
    local_max_only = 1

    # output
    snr = np.zeros_like(data, order='f')
    signal = np.zeros_like(data, order='f')

    peaks_f.peak_snr_filter(data, a, b, c, mask, local_max_only, snr, signal)

    assert(snr[4, 6] == 0)


def test_fortran():

    npx = 100
    npy = 200
    x, _ = np.meshgrid(np.arange(0, npx), np.arange(0, npy), indexing='ij')
    assert(x.shape[0] == npx)
    dat = x**2
    dat = dat.astype(np.float64)
    dat = np.asfortranarray(dat)
    assert(dat[1, 1] == 1)
    mask = np.ones((npx, npy))
    mask = np.asfortranarray(mask)
    out = np.empty_like(dat)
    out = np.asfortranarray(out)
    nin = 0
    ncent = 0
    nout = 1
    # peaks_f.peaker.squarediff(dat, nout, nout, nin, nin)
    peaks_f.peaker.boxsnr(dat, mask, out, nin, ncent, nout)
    noise = np.sqrt(((0+1+16)*3-1)/8. - (((0+1+4)*3-1)/8.)**2)
    sig = 1 - ((0+1+4)*3-1)/8.
    # print(dat[0:3, 0:3])
    # print(out[0:3, 0:3])
    assert(np.abs(out[1, 1] - sig/noise) < 1e-6)


