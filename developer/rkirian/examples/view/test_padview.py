import numpy as np
import reborn
from reborn.simulate.examples import jungfrau4m_pads
from reborn.viewers.qtviews.padviews import PADView

np.random.seed(0)

pads = jungfrau4m_pads()
dats = [np.random.random(p.shape()) for p in pads]
padview = PADView(raw_data=dats, pad_geometry=pads)
padview.start()

