import numpy as np
from scipy.special import lpmn


class ylmIntegration(object):

    dtheta = None  #: Angular spacing of grid points.
    theta = None  #: A :math:`2L+2` array of equally spaced midpoint theta values.
    phi = None  #: A :math:`2L+2` array of equally spaced :math:`\phi` values, starting at :math:`\phi=0`
    w = None  #: A :math:`2L+2` array of weights for the theta points, includes the :math:`d\phi`.

    def __init__(self, L):
        r"""
        Uniform spherical coordinate angular grid and integration weights.  Upon initialization, the following
        attributes are calculated:

        *dtheta* (float): The angular spacing of grid points.

        *theta* (|ndarray|): A :math:`2L+2` array of equally spaced midpoint theta values.

        *phi* (|ndarray|): A :math:`2L+2` array of equally spaced :math:`\phi` values, starting at :math:`\phi=0`

        *w* (|ndarray|): A :math:`2L+2` array of weights for the theta points, includes the :math:`d\phi`.

        Arguments:
            L (int): Maximum :math:`L` for :math:`Y^*_{lm} Y_{lm}` to be correctly integrated.
        """
        self.dtheta = np.pi/(2*L+2)
        self.theta = (np.arange(2*L+2)+0.5)*self.dtheta
        self.phi = np.arange(2*L+2)*2.0*self.dtheta
        self.w = np.zeros(2*L+2, dtype=np.float64)
        self.L = L
        for j in range(2*L+2):
           for i in range(L+1):
              self.w[j] += 4*np.pi/(2*(L+1)**2)*np.sin(((2*j+1)*np.pi)/(4*(L+1)))\
                    *np.sin(((2*j+1)*(2*i+1)*np.pi)/(4*(L+1)))/(2*i+1)
        # calculate Y_lm(theta_j,phi=0) m nonnegative using numpy P_lm
        pml = np.zeros([2*L+2, L+1, L+1], dtype=np.float64)
        self.ylm = np.zeros([2*L+2, L+1, L+1], dtype=np.float64)
        lfac = np.zeros(2*L+1, dtype=np.float64)
        for i in range(1, 2*L+1):
           lfac[i] = lfac[i-1]+np.log(i)
        for j in range(2*L+2):
           pml[j, :, :], _ = lpmn(L, L, np.cos(self.theta[j]))
        for l in range(L+1):
           for m in range(L+1):
              fac = np.sqrt((2*l+1)/(4.0*np.pi)*np.exp(lfac[l-m]-lfac[l+m]))
              self.ylm[:, l, m] = fac*pml[:, m, l]

    def ylmcoef(self, f):
        r"""
        Calculates expansion coefficients :math:`a_{lm}` for the expansion

        .. math:
            f(\theta, \phi) = \sum_{lm} a_{lm} Y_{lm}(\theta, \phi)

        The function :math:`f(\theta, \phi)` must be provided as a numpy array of real or complex values corresponding
        to the self.theta, self.phi coordinates.  The coefficients are returned in an numpy array a_lm[l,m] of shape
        (L+1,2*L+1), with :math:`a_{lm} = 0` for unused :math:`m` values :math:`-L < m < L`.  Python wrap-around
        indexing is used so negative m indices values correspond to coefficients with :math:`-m`.

        Arguments:
            f (|ndarray|): Function evaluated at the coordinates self.theta, self.phi.
        """
        ylmc = np.zeros([self.L+1, 2*self.L+1], dtype=np.complex128)
        ft = np.fft.fft(f)
        for l in range(self.L+1):
            ylmc[l, 0] = np.sum(self.ylm[:, l, 0]*self.w*ft[:, 0])
            fac = -1.0
            for m in range(1, l+1):
                ylmc[l, m] = np.sum(self.ylm[:, l, m]*self.w*ft[:, m])
                ylmc[l, -m] = fac*np.sum(self.ylm[:, l, m]*self.w*ft[:, -m])
                fac = -fac
        return ylmc
