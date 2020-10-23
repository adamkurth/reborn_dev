import numpy as np
import reborn


def plugin(self):
    r""" Plugin for PADView. """
    self.debug('plugin(subtract_median_ss)')
    data = self.get_pad_display_data()
    for i in range(len(data)):
        data[i] -= np.median(data[i], axis=0).reshape((1, data[i].shape[1]))
    self.set_pad_display_data(data, update_display=True, percentiles=(2, 98), colormap='bipolar')
