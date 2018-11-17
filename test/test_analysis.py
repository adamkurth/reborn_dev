import sys
sys.path.append('..')
import numpy as np
from bornagain.analysis.peaks_f import peak_snr_filter, squarediff


def test_peak_snr_filter():
    # input
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

    peak_snr_filter(data, a, b, c, mask, local_max_only, snr, signal)

    assert(snr[4, 6] == 0)


def test_peaker():

    npx = 100
    npy = 200
    p = np.random.rand(npx, npy)
    p0 = p.copy('f')
    nin = 20
    nout = 30
    squarediff(p0, npx, npy, nout, nout, nin, nin)


