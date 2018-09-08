import sys
sys.path.append('../..')

from bornagain.simulate import examples
from bornagain.viewers.qtviews import PADView
import numpy as np
import pyqtgraph as pg

sim = examples.lysozyme_molecule()

I = [np.log10(d*0.001+10) for d in sim['intensity']]
padgui = PADView(data=I, pad_geometry=sim['pad_geometry'])
# padgui.add_rings([200, 400, 600, 800], pens=[pg.mkPen([255, 0, 0], width=2)]*4)
# padgui.enable_geometry_adjustment()
padgui.start()