import numpy as np
from numba import jit

real_t = np.float64
complex_t = np.complex128

@jit(nopython=True)
def phase_factor_qrf_numba(q, r, f, R, a, add):

    n_pixels = q.shape[0]
    n_atoms = r.shape[0]

    if add == 0:
        a[:] = 0

    for i in range(n_pixels):
        qi = q[i, :]

        rqi = np.array([R[0, 0] * qi[0] + R[0, 1] * qi[1] + R[0, 2] * qi[2],
                        R[1, 0] * qi[0] + R[1, 1] * qi[1] + R[1, 2] * qi[2],
                        R[2, 0] * qi[0] + R[2, 1] * qi[1] + R[2, 2] * qi[2]])
        ai = 0j
        for j in range(n_atoms):
            rj = r[j, :]
            ph = -1j*np.sum(rqi*rj)
            ai += f[j]*np.exp(ph)
        a[i] = ai

    return a


def phase_factor_qrf(q, r, f, R=None, a=None, add=False):

    q = q.astype(real_t)
    r = r.astype(real_t)
    f = f.astype(complex_t)
    if R is None:
        R = np.eye(3)
    R = R.astype(real_t)
    if a is None:
        a = np.empty(q.shape[0], dtype=complex_t)
    if add is False:
        add = 0
    else:
        add = 1

    return phase_factor_qrf_numba(q, r, f, R, a, add)
