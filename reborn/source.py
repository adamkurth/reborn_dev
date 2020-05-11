r"""
Classes related to x-ray sources.
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np
from scipy import constants as const

hc = const.h*const.c  # pylint: disable=invalid-name


class Beam():

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

    def __init__(self, beam_vec=np.array([0, 0, 1]), photon_energy=1.602e-15, wavelength=None,
                 polarization_vec=np.array([1, 0, 0]), pulse_energy=1e-3, diameter_fwhm=1e-6):

        self.beam_vec = beam_vec
        self.polarization_vec = polarization_vec

        self.photon_energy = photon_energy

        if photon_energy is None:
            self.wavelength = wavelength

        self.diameter_fwhm = diameter_fwhm
        self.pulse_energy = pulse_energy

    def __str__(self):
        out = ''
        out += 'beam_vec: %s\n' % self.beam_vec.__str__()
        out += 'polarization_vec: %s\n' % self._polarization_vec.__str__()
        out += 'wavelength: %s\n' % self.wavelength.__str__()
        out += 'photon_number_fluence: %s\n' % self.photon_number_fluence.__str__()
        return out

    @property
    def hash(self):
        return hash(self.__str__())

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
        r""" The principle polarization vector :math:`\hat{u}`.  This should be orthogonal to the incident beam
        direction.  The complementary polarization vector is :math:`\hat{u}\times\hat{b}`"""
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
        r""" Photon wavelength in meters."""
        return hc/self.photon_energy

    @wavelength.setter
    def wavelength(self, value):
        self.photon_energy = hc/value

    @property
    def pulse_energy(self):
        r""" Pulse energy in J."""
        return self._pulse_energy

    @pulse_energy.setter
    def pulse_energy(self, val):
        self._pulse_energy = val

    @property  # this cannot be set - set pulse energy instead
    def n_photons(self):
        r""" Number of photons per pulse."""
        return self.pulse_energy / self.photon_energy

    @property
    def fluence(self):
        r""" Same as energy_fluence.  Don't use this method."""
        return self.energy_fluence

    @property
    def photon_number_fluence(self):
        r""" Pulse fluence in photons/m^2."""
        return self.n_photons/(np.pi * self.diameter_fwhm**2 / 4.0)

    @property
    def energy_fluence(self):
        r""" Pulse fluence in J/m^2."""
        return self.pulse_energy/(np.pi * self.diameter_fwhm**2 / 4.0)
