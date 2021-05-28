import numpy as np
from reborn.external import pyqtgraph as myqtgraph

im = np.arange(10*15)
im = im.reshape((10, 15))
myqtgraph.image(im, view='plot', ss_lims=(0, 1), fs_lims=(0, 15))
myqtgraph.keep_open()
