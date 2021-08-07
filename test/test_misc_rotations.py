import numpy as np
import scipy
from reborn.misc.rotate import Rotate3D, Rotate3Dvkfft, have_gpu


def makegaussians(w, g, s, n):
    x = np.linspace(-0.5 * (1.0 - 1.0 / n), 0.5 * (1.0 - 1.0 / n), num=n)
    d = np.zeros((n, n, n), dtype=np.float64)
    for ig in range(len(w)):
        xgauss = np.exp(-0.5 * ((x - g[ig, 0]) / s) ** 2)
        ygauss = np.exp(-0.5 * ((x - g[ig, 1]) / s) ** 2)
        zgauss = np.exp(-0.5 * ((x - g[ig, 2]) / s) ** 2)
        d += w[ig] * np.tile(np.reshape(xgauss, (n, 1, 1)), (1, n, n)) * \
             np.tile(np.reshape(ygauss, (1, n, 1)), (n, 1, n)) * \
             np.tile(np.reshape(zgauss, (1, 1, n)), (n, n, 1))
    return d


rng = np.random.default_rng(1717171717)
Nr = 10
Rs = scipy.spatial.transform.Rotation.random(Nr, random_state=rng)
Ngr = 8
Ngi = 8
sigma = 0.05
gr0 = (rng.random((Ngr, 3)) - 0.5) * sigma * 3.0
wr = rng.random(Ngr) - 0.5
gi0 = (rng.random((Ngi, 3)) - 0.5) * sigma * 3.0
wi = rng.random(Ngi) - 0.5
N = 32
methods = [Rotate3D]  # , rotate3Dl, rotate3Dj]
# if have_gpu:
#     methods.append(Rotate3Dvkfft)
types = [np.complex128, np.complex64, np.float64, np.float32]


def test_01():
    gr = gr0.copy()
    gi = gi0.copy()
    datar = makegaussians(wr, gr, sigma, N)
    data = datar + 1j * makegaussians(wi, gi, sigma, N)
    for t in types:
        for m in methods:
            gr = gr0.copy()
            gi = gi0.copy()
            if t == np.complex128 or t == np.complex64:
                r3df = m(data.astype(t))
                d_max = np.max(np.abs(data))
            else:
                r3df = m(datar.astype(t))
                d_max = np.max(np.abs(datar))
            for ir in Rs:
                gr = ir.apply(gr)
                gi = ir.apply(gi)
                r3df.rotation(ir)
                if t == np.complex128 or t == np.complex64:
                    error = np.max(np.abs(r3df.f - makegaussians(wr, gr, sigma, N)
                                          - 1j * makegaussians(wi, gi, sigma, N))) / d_max
                    assert (error < 1e-4)  # Weak tests; only checks if something is horribly wrong
                else:
                    error = np.max(np.abs(r3df.f - makegaussians(wr, gr, sigma, N))) / d_max
                    assert (error < 1e-4)  # Weak test
