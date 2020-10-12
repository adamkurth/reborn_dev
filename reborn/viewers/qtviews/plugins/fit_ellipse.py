import os
import numpy as np
import reborn
from reborn.analysis.optimize import fit_ellipse_pad


def plugin(self):
    r""" Plugin for PADView. """
    self.debug(os.path.basename(__file__))
    efit = fit_ellipse_pad(self.pad_geometry, self.mask_data, threshold=0.5)
    r = np.mean(efit[6:8])
    a = efit[6]
    b = efit[7]
    X0 = efit[8]
    Y0 = efit[9]
    theta = -efit[10]
    phi = np.arange(1000)*2*np.pi/(999)
    X = a*np.cos(phi)
    Y = b*np.sin(phi)
    # x = (X - X0)*np.cos(theta) + (Y - Y0)*np.sin(theta)
    # y = -(X - X0)*np.sin(theta) + (Y - Y0)*np.cos(theta)
    x = X*np.cos(theta) + Y*np.sin(theta) + X0
    y = -X*np.sin(theta) + Y*np.cos(theta) + Y0
    self.add_scatter_plot(x, y)
    self.add_rings([r])
