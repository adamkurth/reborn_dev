import numpy as np
import reborn


def plugin(self):
    r""" Plugin for PADView. """
    profiler = reborn.detector.RadialProfiler(pad_geometry=self.pad_geometry, beam=self.beam)
    data = profiler.subtract_median_profile(self.get_pad_display_data(), mask=self.mask_data)
    self.set_pad_display_data(data, auto_levels=True, update_display=True)
