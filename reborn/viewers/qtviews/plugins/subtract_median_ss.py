import numpy as np

def plugin(self):
    r""" Plugin for PADView. """
    self.debug('plugin(subtract_median_ss)')
    pad_data = self.get_pad_display_data()
    for i in range(len(pad_data)):
        p = pad_data[i]
        print(p.shape)
        s = np.median(p, axis=1).reshape((p.shape[0], 1))
        print(s.shape)
        p -= s
        pad_data[i] = p
    if self.processed_data is None:
        self.processed_data = {}
    self.processed_data['pad_data'] = pad_data
    self.update_pads()
