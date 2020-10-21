import numpy as np
import reborn


def plugin(self):
    r""" Plugin for PADView. """
    self.debug('plugin(subtract_median_ss)')
    data = self.get_pad_display_data()
    for i in range(len(data)):
        data[i] -= np.median(data[i], axis=1).reshape((data[i].shape[0], 1))
    self.set_pad_display_data(data, update_display=True)
    d = reborn.detector.concat_pad_data(data)
    upper = np.percentile(d, 98)
    lower = np.percentile(d, 2)
    lim = max(np.abs(upper), np.abs(lower))
    self.set_preset_colormap('bipolar')
    self.set_levels(-lim, lim)
