r"""
Classes related to x-ray sources.
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np
from bornagain.units import hc


class Beam(object):

    r"""
    A minimal containor to gather x-ray beam properties.
    """

    # derived quantities:
    # wavelength (from photon_energy)
    # n_photons (from pulse energy and photon energy)
    # fluence (from pulse_energy and beam_diameter_fwhm)


    beam_profile = 'tophat'
    beam_vec = None  #: Nominal direction of incident beam
    polarization_vec = None
    polarization_weight = 1  #: Weight of the first polarization vector
    wavelength = None  #: Nominal photon wavelength
    spectral_width_fwhm = 0
    pulse_energy = None
    beam_divergence_fwhm = 0
    diameter_fwhm = 0
    fluence_jitter_fwhm = 0

    def __init__(self, beam_vec=np.array([0, 0, 1]), photon_energy=None, wavelength=None,
                 polarization_vec=np.array([1, 0, 0]), pulse_energy=None, diameter_fwhm=None):

        self.beam_vec = beam_vec
        self.polarization_vec = polarization_vec

        self.photon_energy = photon_energy

        if photon_energy is None:
            self.wavelength = wavelength

        self.diameter_fwhm = diameter_fwhm
        self.pulse_energy = pulse_energy

    @property
    def beam_vec(self):
        return self._beam_vec

    @beam_vec.setter
    def beam_vec(self, vec):
        self._beam_vec = np.array(vec)

    @property
    def wavelength(self):
        if self.photon_energy is not None:
            return hc/self.photon_energy
        else:
            return None

    @wavelength.setter
    def wavelength(self, value):
        if value is not None:
            self.photon_energy = hc/value

    @property
    def n_photons(self):
        pulse = self.pulse_energy
        photon = self.photon_energy
        if pulse is not None and photon is not None:
            return pulse/photon
        else:
            return None

    @property
    def fluence(self):
        if self.pulse_energy is not None and self.diameter_fwhm is not None:
            return self.pulse_energy/(np.pi * self.diameter_fwhm**2 / 4.0)
        else:
            return None









