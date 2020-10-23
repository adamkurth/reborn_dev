import numpy as np
from reborn import detector, source


def plugin(self):
    r""" Plugin for PADView. """
    data = detector.concat_pad_data(self.get_pad_display_data())
    data /= detector.concat_pad_data([p.polarization_factors(beam=self.beam) for p in self.pad_geometry])
    self.set_pad_display_data(data, auto_levels=True, update_display=True)
