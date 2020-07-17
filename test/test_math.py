import numpy as np
from scipy.special import sph_harm
from reborn.math.spherical import ylmIntegration


def test_ylmIntegration():
    small = 1.0e-12
    L = 15
    f = np.zeros([2 * L + 2, 2 * L + 2], dtype=np.complex128)
    intylm = ylmIntegration(L)
    for l in range(L + 1):
        for m in range(-l, l + 1):
            for j in range(2 * L + 2):
                f[j, :] = sph_harm(m, l, intylm.phi[:], intylm.theta[j])
            ylmc = intylm.calc_ylmcoef(f)
            ylmc[l, m] -= 1.0
            assert (np.max(np.abs(ylmc)) < small)
