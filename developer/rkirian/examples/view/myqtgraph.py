import numpy as np
from reborn.external import pyqtgraph as myqtgraph

d = np.arange(10*15)
d = d.reshape((10, 15))
im = myqtgraph.image(d, view='plot', ss_lims=(0, 1), fs_lims=(0, 15))

myqtgraph.keep_open()
