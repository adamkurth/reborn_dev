import numpy as np
from reborn.analysis.optimize import fit_ellipse_pad


def plugin(self):
    r""" Plugin for PADView. """
    self.debug('ellipse_fit')
    efit = fit_ellipse_pad(self.pad_geometry, self.mask_data, threshold=1)
    r = np.mean(efit[6:8])
    print(efit)
    self.add_rings([r*self._scale_factor(0)])
