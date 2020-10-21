import numpy as np
from reborn import detector, source


def plugin(self):
    r""" Plugin for PADView. """
    data = detector.concat_pad_data(self.get_pad_display_data())
    pads = self.pad_geometry
    sang = detector.concat_pad_data([p.solid_angles() for p in pads])
    data /= sang*1e6
    self.set_pad_display_data(data, auto_levels=True, update_display=True)
