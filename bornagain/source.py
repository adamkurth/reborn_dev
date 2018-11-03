"""
Classes related to x-ray sources.
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np

class Beam(object):
    r"""
    A minimal containor to gather x-ray beam properties.
    """

    wavelength = None  #: Nominal photon wavelength
    beam_vec = None  #: Nominal direction of incident beam
    polarization_vec = None  #: Direction of the first polarization vector
    polarization_weight = 1  #: Weight of the first polarization vector
    photon_energy = 0  # This should be a function
    spectral_width_fwhm = 0
    pulse_energy = 0
    n_photons = 0
    beam_divergence_fwhm = 0
    beam_diameter_fwhm = 0
    fluence = 0
    fluence_jitter_fwhm = 0

    def __init__(self, beam_vec=np.array([0, 0, 1]), wavelength=1.5e-10, polarization_vec=np.array([1, 0, 0])):

        self.beam_vec = beam_vec
        self.polarization_vec = polarization_vec
        self.wavelength = wavelength
