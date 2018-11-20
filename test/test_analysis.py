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
    p = np.random.rand(npx, npy)
    p0 = np.asfortranarray(p) #.copy('f')
    nin = 20
    nout = 30
    peaks_f.peaker.squarediff(p0, nout, nout, nin, nin)
