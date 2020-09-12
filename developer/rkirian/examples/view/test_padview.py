import numpy as np
import reborn
from reborn.simulate import examples
from reborn.viewers.qtviews.padviews import PADView

np.random.seed(0)


pads = reborn.detector.tiled_pad_geometry_list(pad_shape=(20, 15), pixel_size=1000e-6, distance=0.1, tiling_shape=(2, 2), pad_gap=0)
# pads = jungfrau4m_pads()
# pads = examples.cspad_pads(); pads = pads[0:12]
dats = [np.random.random(p.shape()) for p in pads]
for d in range(len(dats)):
    dats[d] += d
    dats[d][0:5, 0:10] = -1
for d in dats:
    x, y = np.indices(d.shape)
    d -= x/100
padview = PADView(raw_data=dats, pad_geometry=pads, debug_level=0)
padview.show_pad_labels()
padview.show_coordinate_axes()
padview.set_title('Title')
padview.start()
