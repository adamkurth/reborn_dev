"""
Under dev, this is a simple class for viewing PAD geometries
"""

import numpy as np
import sys
try:
    import matplotlib
    import pylab as plt
except ImportError:
    pass
from bornagain import units
from bornagain.detector import PADGeometry

class SimplePAD(PADGeometry):

    """
    A simple child class to PADGeometry with some higher level functionality

    This will return a detector object representing a
    square pixel array detector

    .. note::
        - One can readout pixel intensities using :func:`readout`
        - After reading out amplitudes, one can display pixels using :func:`display`

    Arguments
        - n_pixels (int)
            the number of pixels along one edge

        - pixsize (float)
            the edge length of the square pixels in meters

        - detdist (float)
            the distance from the interaction region to the point where
            the forward beam intersects the detector (in meters)

        - wavelen (float)
            the wavelength of the photons (in Angstroms)

        - center (tuple)
            the fast-scan center coordinate and the slow-scan center coordinate
            Not used now
    """

    def __init__(self, n_pixels=1000, pixsize=0.00005, detdist=0.05, wavelen=1e-10, center=None,
                 *args, **kwargs):

        """
        A simple wrapper to bornagain PADGeometry that includes a
        simple viewer

        Parameters
        ==========
        n_pixels, int
            number of pixels along each dimension

        detdist, float
            shortest distance from interaction region to detector

        wavelen, float

        center, tuple
            the fast-scan, slow-scan coordinate of the beam center
            Not used currently..

        args and kwargs passed to The bornagain PAGeometry method

        Returns
        =======
        SimplePAD object with a simple viewing method

        """
        n_pixels = int( n_pixels)
        PADGeometry.__init__(self, *args, **kwargs)

        self.detector_distance = detdist
        self.wavelength = wavelen
        self.si_energy = units.hc / wavelen

        self.simple_setup(n_pixels=n_pixels,
                          pixel_size=pixsize,
                          distance=detdist)

        self.fig = None

        # shape of the 2D det panel (2D image)
        self.img_sh = self.shape()

        if center is not None:
            assert (len(center) == 2)
            assert (center[0] < self.n_fs)
            assert (center[1] < self.n_ss)
            self.center = center
        else:
            self.center = map(lambda x: x / 2., self.img_sh)

        self.SOLID_ANG = self.solid_angles()

        self._make_qmag()

        # useful functions fr converting between pixel radii and momentum transfer
        self.rad2q = lambda rad: 4 * np.pi * np.sin(.5 * np.arctan(rad * pixsize / detdist)) / wavelen
        self.q2rad = lambda q: np.tan(np.arcsin(q * wavelen / 4 / np.pi) * 2) * detdist / pixsize

        self.intens = None

    def _make_qmag(self):

        r"""
        Makes the momentum transfer of each Q
        """

        self.Q_vectors = self.q_vecs(
            beam_vec=np.array([0, 0, 1]),
            wavelength=self.wavelength)
        self.Qmag = np.sqrt(np.sum(self.Q_vectors ** 2, axis=1))

    def readout(self, amplitudes):
        """
        Given scattering amplitudes, this calculates the corresponding intensity values.

        Arguments
            amplitudes (complex np.ndarray) : Scattering amplitudes same shape as `self.Q`

        Returns
            np.ndarray : Scattering intensities as a 2-D image.
        """
        self.intens = (np.abs(amplitudes) ** 2).reshape(self.img_sh)
        return self.intens

    def readout_finite(self, amplitudes, qmin, qmax, flux=1e20):
        """
        Get scattering intensities as a 2D image considering
        finite scattered photons

        Arguments:
            amplitudes (complex np.ndarray) : Scattering amplitudes same shape as `self.Q`.
            qmin (float) : Minimum q to generate intensities
            qmax (float) : Maximum q to generate intenities
            flux (float) : Forward beam flux in Photons per square centimeter

        Returns:
            np.ndarray : Scattering intensities as a 2-D image.
        """
        self.intens = (np.abs(amplitudes) ** 2).reshape(self.img_sh)
        struct_fact = (np.abs(amplitudes) ** 2).astype(np.float64)

        if qmin < self.Qmag.min():
            qmin = self.Qmag.min()
        if qmax > self.Qmag.max():
            qmax = self.Qmag.max()

        ilow = np.where(self.Qmag < qmin)[0]
        ihigh = np.where(self.Qmag > qmax)[0]

        if ilow.size:
            struct_fact[ilow] = 0
        if ihigh.size:
            struct_fact[ihigh] = 0

        rad_electron = 2.82e-13  # cm
        phot_per_pix = struct_fact * self.SOLID_ANG * flux * rad_electron ** 2
        total_phot = int(phot_per_pix.sum())

        pvals = struct_fact / struct_fact.sum()

        self.intens = np.random.multinomial(total_phot, pvals)

        self.intens = self.intens.reshape(self.img_sh)

        return self.intens

    def display(self, use_log=True, vmax=None, pause=None, **kwargs):
        """
        Displays a detector. Extra kwargs are passed
        to matplotlib.figure

        .. note::
            - Requires matplotlib.
            - Must first run :func:`readout` or :func:`readout_finite`
                at least one time

        Arguments
            - use_log (bool)
                whether to use log-scaling when displaying the intensity image.

            - vmax (float)
                colorbar scaling argument.
        """

        assert (self.intens is not None)

        if 'matplotlib' not in sys.modules:
            print("You need matplotlib to plot!")
            return
        # plt = matplotlib.pylab

        if self.fig is None:
            fig = plt.figure(**kwargs)
        else:
            fig = self.fig
        fig.clear()
        ax = plt.gca()
        qx_min, qy_min = self.Q_vectors[:, :2].min(0)
        qx_max, qy_max = self.Q_vectors[:, :2].max(0)
        extent = (qx_min, qx_max, qy_min, qy_max)
        if use_log:
            ax_img = ax.imshow(
                np.log1p(
                    self.intens),
                extent=extent,
                cmap='gnuplot',
                interpolation='nearest')
            cbar = fig.colorbar(ax_img)
            cbar.ax.set_ylabel('log(photon counts)', rotation=270, labelpad=12)
        else:
            assert (vmax is not None)
            ax_img = ax.imshow(
                self.intens,
                extent=extent,
                cmap='gnuplot',
                interpolation='nearest',
                vmax=vmax)
            cbar = fig.colorbar(ax_img)
            cbar.ax.set_ylabel('photon counts', rotation=270, labelpad=12)

        ax.set_xlabel(r'$q_x\,\,\AA^{-1}$')
        ax.set_ylabel(r'$q_y\,\,\AA^{-1}$')

        if pause is None:
            plt.show()
        elif pause is not None and self.fig is None:
            self.fig = fig
            plt.draw()
            plt.pause(pause)
        else:
            plt.draw()
            plt.pause(pause)

