import numpy as np
import reborn


def plugin(self):
    r""" Plugin for PADView. """
    self.debug('plugin(subtract_median_ss)')
    pad_data = self.get_pad_display_data()
    for i in range(len(pad_data)):
        pad_data[i] -= np.median(pad_data[i], axis=1).reshape((pad_data[i].shape[0], 1))
    if self.processed_data is None:
        self.processed_data = {}
    self.processed_data['pad_data'] = pad_data
    self.update_pads()
    d = reborn.detector.concat_pad_data(pad_data)
    upper = np.percentile(d, 98)
    lower = np.percentile(d, 2)
    lim = max(np.abs(upper), np.abs(lower))
    self.set_preset_colormap('bipolar')
    self.update_pads()
    self.set_levels(-lim, lim)
    self.process_events()
