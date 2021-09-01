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
from scipy.special import lpmn


class ylmIntegration(object):
    r""" Class for calculating coefficients :math:`a_{lm}` in the spherical harmonics expansion

    .. math::

        f(\theta, \phi) = \sum_{lm} a_{lm} Y_{lm}(\theta, \phi) ;.

    The spherical harmonics :math:`Y_{lm}(\theta, \phi)` are the same as the |scipy| function
    :func:`scipy.special.sph_harm`, HOWEVER, the :math:`\theta` and :math:`\phi` variables in the |scipy| documentation
    are switched when compared to the common physics convention used here.  The normalization is defined so that
    setting :math:`f(\theta, \phi) = Y_{l'm'}(\theta, \phi)` you will result in

    .. math::

        a_{lm} = \delta(l-l')\delta(m-m')
    """
    # Note that if we define attributes here (rather than in the __init__ function), they will appear in the HTML
    # documentation.  The '#:' below is used to get additional text in the HTML docs.
    theta = None  #: A :math:`2L+2` |ndarray| of equally spaced midpoint :math:`\theta` values.
    phi = None  #: A :math:`2L+2` |ndarray| of equally spaced :math:`\phi` values, starting at :math:`\phi=0`

    def __init__(self, L):
        r"""
        The initialization routine calculates and stores the angular spacing :math:`d\theta = \pi/(2L+2)`, the
        :math:`2L+2` equally spaced midpoint :math:`\theta` values (we exclude the poles :math:`\theta=0` and
        :math:`\theta= \pi`), the :math:`2L+2` array of equally spaced :math:`\phi` values (starting at :math:`\phi=0`).
        Additional quantities (e.g. integration weights) are also calculated and stored.

        Arguments:
            L (int): Maximum :math:`L` for :math:`Y^*_{lm} Y_{lm}` to be correctly integrated.
        """
        self.dtheta = np.pi / (2 * L + 2)
        self.theta = (np.arange(2 * L + 2) + 0.5) * self.dtheta
        self.phi = np.arange(2 * L + 2) * 2.0 * self.dtheta
        self.w = np.zeros(2 * L + 2, dtype=np.float64)
        self.L = L
        for j in range(2 * L + 2):
            for i in range(L + 1):
                self.w[j] += 4 * np.pi / (2 * (L + 1) ** 2) * np.sin(((2 * j + 1) * np.pi) / (4 * (L + 1))) \
                             * np.sin(((2 * j + 1) * (2 * i + 1) * np.pi) / (4 * (L + 1))) / (2 * i + 1)
        # calculate Y_lm(theta_j,phi=0) m nonnegative using numpy P_lm
        pml = np.zeros([2 * L + 2, L + 1, L + 1], dtype=np.float64)
        self.ylm = np.zeros([2 * L + 2, L + 1, L + 1], dtype=np.float64)
        lfac = np.zeros(2 * L + 1, dtype=np.float64)
        for i in range(1, 2 * L + 1):
            lfac[i] = lfac[i - 1] + np.log(i)
        for j in range(2 * L + 2):
            pml[j, :, :], _ = lpmn(L, L, np.cos(self.theta[j]))
        for l in range(L + 1):
            for m in range(L + 1):
                fac = np.sqrt((2 * l + 1) / (4.0 * np.pi) * np.exp(lfac[l - m] - lfac[l + m]))
                self.ylm[:, l, m] = fac * pml[:, m, l]

    def calc_ylmcoef(self, f):
        r"""
        Calculates expansion coefficients :math:`a_{lm}`.  The function :math:`f(\theta, \phi)` must be provided as an
        |ndarray| of real or complex values corresponding to the stored grid of :math:`\theta`, :math:`\phi`
        coordinates.  The coefficients :math:`a_{lm}` are returned in an |ndarray| of shape (:math:`L+1`, :math:`2L+1`),
        with :math:`a_{lm} = 0` for unused :math:`m` values :math:`-L < m < L`.  Numpy wrap-around indexing is used so
        that negative :math:`m` array indices correspond to coefficients with negative indices.

        Arguments:
            f (|ndarray|): Function to be expanded in spherical harmonics, evaluated at the :math:`\theta`, :math:`\phi`
                           coordinates, which may be accessed via the :attr:`~theta`, :attr:`~phi` attributes.

        Returns:
            |ndarray|: :math:`a_{lm}` coefficients.
        """
        ylmc = np.zeros([self.L + 1, 2 * self.L + 1], dtype=np.complex128)
        ft = np.fft.fft(f)
        for l in range(self.L + 1):
            ylmc[l, 0] = np.sum(self.ylm[:, l, 0] * self.w * ft[:, 0])
            fac = -1.0
            for m in range(1, l + 1):
                ylmc[l, m] = np.sum(self.ylm[:, l, m] * self.w * ft[:, m])
                ylmc[l, -m] = fac * np.sum(self.ylm[:, l, m] * self.w * ft[:, -m])
                fac = -fac
        return ylmc

        # .. math::
        #
        #     w_j=\sum_i\frac{4\pi}{2(2i+1)(L+1)^2}\sin\left(\frac{(2j+1)\pi}{4(L+1)}\right)\sin\left(\frac{(2j+1)(2i+1)\pi}{4(L+1)}\right)