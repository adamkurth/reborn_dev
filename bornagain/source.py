"""
Classes related to x-ray sources.
"""

import numpy as np
import utils


class Beam(object):

    def __init__(self):

        self.wavelength = None
        self.B = np.array([0, 0, 1])
        self.P = np.array([1, 0, 0])
        self.polarization_vectors = [np.array([1.0, 0, 0]),np.array([0, 1.0, 0])]
        self.polarization_weights = [1.0,0]
        self.photon_energy = 0
        self.spectral_width_fwhm = 0
        self.pulse_energy = 0
        self.n_photons = 0
        self.divergence_fwhm = 0
        self.beam_diameter_fwhm = 0
        self.mean_fluence = 0
        self.fluence_jitter_fwhm = 0

    def __str__(self):

        s = ""
        if self.wavelength is not None:
            s += "wavelength : %g\n" % self.wavelength
        else:
            s += "wavelength : None\n"
        s += "B : [%g, %g, %g]\n" % (self.B[0], self.B[1], self.B[2])
        return s

    def check(self):

        if self.wavelength <= 0:
            utils.warn("Bad wavelength: %f" % self.wavelength)
            return False
        return True
