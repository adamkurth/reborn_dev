import numpy as np
import reborn


def plugin(self):
    r""" Plugin for PADView. """
    self.debug('ellipse_fit')
    efit = reborn.analysis.optimize.fit_ellipse_pad(self.pad_geometry, self.mask_data, threshold=1)
    r = np.mean(efit[6:8])
    print(efit)
    self.add_rings([r*self._scale_factor(0)])
