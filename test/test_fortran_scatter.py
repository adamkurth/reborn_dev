import numpy as np
from reborn import fortran


def debye(r_vecs, q_mags, f_idx, ff):
    """ Simple pure-Python Debye formula"""
    nq = q_mags.shape[0]
    qp = q_mags / np.pi
    na = r_vecs.shape[0]
    tot = np.zeros(nq, dtype=np.complex128)
    for ri in range(na):
        r1 = r_vecs[ri, :]
        f1 = ff[f_idx[ri], :]
        # print('f1',f1)
        tot += f1*np.conj(f1)  # Diagonal terms ij = ii
        for rj in range(ri+1, na):
            r2 = r_vecs[rj, :]
            f2 = ff[f_idx[rj], :]
            # print('f2',f2)
            rij = np.sqrt(np.sum(r1**2 + r2**2))
            qr = qp * rij
            tot += 2*np.real(f1*np.conj(f2))*np.sinc(qr)
    return tot


def test_01():
    q = np.array([0, 1, 2], dtype=np.float64)*1e10
    f = np.array([[4+1j, 3+1j, 2+1j], [3+1j, 2+2j, 1+3j]], dtype=np.complex128)
    r = np.zeros((2, 3), dtype=np.float64)
    r[0, 0] = 1e-10
    fidx = np.array([0, 1], dtype=np.int16)
    a = np.zeros(3, dtype=np.float64)
    fortran.scatter_f.debye(r.T, q, fidx, f.T, a)
    aa = debye(r, q, fidx, f)
    # Check forward scatter
    assert np.abs(a[0] - np.real((f[0, 0]+f[1, 0])*np.conj(f[0, 0]+f[1, 0]))) < 1e-6
    assert np.abs(aa[0] - np.real((f[0, 0] + f[1, 0]) * np.conj(f[0, 0] + f[1, 0]))) < 1e-6
    # Check all others
    assert np.max(np.abs(a-aa)) < 1e-6
