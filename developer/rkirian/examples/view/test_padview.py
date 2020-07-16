import numpy as np
from reborn.simulate.examples import jungfrau4m_pads
from reborn.viewers.qtviews.padviews import PADView

np.random.seed(0)

pads = jungfrau4m_pads()
dats = [np.random.random(p.shape()) for p in pads]
for d in dats:
    x, y = np.indices(d.shape)
    d -= x/100
padview = PADView(raw_data=dats, pad_geometry=pads, debug_level=1)
padview.set_title('Title')
padview.show()
