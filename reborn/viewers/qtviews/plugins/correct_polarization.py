import numpy as np
from reborn import detector, source


def plugin(self):
    r""" Plugin for PADView. """
    data = detector.concat_pad_data(self.get_pad_display_data())
    pads = self.pad_geometry
    if self.beam is None:
        self.debug('No beam is configured.  Totally guessing that the beam is 9keV photon energy.')
        photon_energy = 9*1.602e-19*1e3
        print('photon_energy', photon_energy)
        self.beam = source.Beam(photon_energy=photon_energy)
    beam = self.beam
    pfac = detector.concat_pad_data([p.polarization_factors(beam=beam) for p in pads])
    data /= pfac
    data = detector.split_pad_data(pads, data)
    if self.processed_data is None:
        self.processed_data = {}
    self.processed_data['pad_data'] = data
    self.update_pads()
    self.set_levels_by_percentiles()