r"""
Classes related to x-ray sources.
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np
from scipy import constants as const

hc = const.h*const.c


class Beam(object):

    r"""
    A minimal containor to gather x-ray beam properties.
    """

    # derived quantities:
    # wavelength (from photon_energy)
    # n_photons (from pulse energy and photon energy)
    # fluence (from pulse_energy and beam_diameter_fwhm)

    photon_energy = None
    _beam_profile = 'tophat'
    _beam_vec = None  #: Nominal direction of incident beam
    _polarization_vec = None
    _polarization_weight = 1  #: Weight of the first polarization vector
    _wavelength = None  #: Nominal photon wavelength
    photon_energy_fwhm = 0
    _pulse_energy = None
    divergence_fwhm = 0
    diameter_fwhm = None
    pulse_energy_fwhm = 0

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
    def beam_profile(self):
        r""" In the future this will be a means of specifying the profile of the incident x-rays.  The only option is
         'tophat' for the time being.  Possibly in the future we could allow for complex wavefronts.  """
        return self._beam_profile

    @beam_profile.setter
    def beam_profile(self, val):
        if val not in ['tophat']:
            raise ValueError("beam.beam_profile must be 'tophat' or ... that's all for now...")
        self._beam_profile = val

    @property
    def beam_vec(self):
        r""" The nominal direction of the incident x-ray beam. """
        return self._beam_vec

    @beam_vec.setter
    def beam_vec(self, vec):
        self._beam_vec = np.array(vec)

    @property
    def polarization_vec(self):
        r""" The principle polarization vector :math:`\hat{u}`.  This should be orthogonal to the incident beam direction.  The
        complementary polarization vector is :math:`\hat{u}\times\hat{b}`"""
        return self._polarization_vec

    @polarization_vec.setter
    def polarization_vec(self, vec):
        self._polarization_vec = np.array(vec)

    @property
    def polarization_weight(self):
        r""" The fraction of f of energy that goes into the principle polarization vector specified by the
        polarization_vec attriute.  The fraction of the energy in the complementary polarization is of course (1-f). """
        return self._polarization_weight

    @polarization_weight.setter
    def polarization_weight(self, val):
        self._polarization_weight = val

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
    def pulse_energy(self):
        if self._pulse_energy is None:
            raise ValueError("beam.pulse_energy has not been defined.  There is no default.")
        else:
            return self._pulse_energy

    @pulse_energy.setter
    def pulse_energy(self, val):
        self._pulse_energy = val

    @property  # this cannot be set - set pulse energy instead
    def n_photons(self):
        return self.pulse_energy / self.photon_energy

    @property
    def fluence(self):
        return self.energy_fluence

    @property
    def photon_number_fluence(self):
        return self.n_photons/(np.pi * self.diameter_fwhm**2 / 4.0)

    @property
    def energy_fluence(self):
        return self.pulse_energy/(np.pi * self.diameter_fwhm**2 / 4.0)
