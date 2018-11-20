import sys
sys.path.append('..')

import numpy as np

from bornagain.viewers.qtviews import PADView
from bornagain.fileio.getters import CheetahFrameGetter
from bornagain.analysis import peaks

geom_file_name = 'data/cxin5016-oy-v1.geom'
cxi_file_name = 'data/cxilu5617-r0149-c00.cxi'

frame_getter = CheetahFrameGetter(cxi_file_name, geom_file_name)
dat = frame_getter.get_frame(0)
pads = dat['pad_data']
shape = pads[0].shape
masks = [np.ones_like(d) for d in pads]
padview = PADView(frame_getter=frame_getter, mask_data=masks)

pfind = peaks.PeakFinderV1(shape=masks[0].shape)
# padview.data_filters = pfind.snr_filter
padview.data_filters = peaks.snr_filter_fortran
padview.add_roi()
padview.start()