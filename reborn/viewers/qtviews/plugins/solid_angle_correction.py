import numpy as np
from reborn import detector, source


def plugin(self):
    r""" Plugin for PADView. """
    data = detector.concat_pad_data(self.get_pad_display_data())
    pads = self.pad_geometry
    sang = detector.concat_pad_data([p.solid_angles() for p in pads])
    data /= sang*1e6
    data = detector.split_pad_data(pads, data)
    if self.processed_data is None:
        self.processed_data = {}
    self.processed_data['pad_data'] = data
    self.update_pads()
    self.set_levels_by_percentiles()