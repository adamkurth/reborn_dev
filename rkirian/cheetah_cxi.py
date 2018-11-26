import sys
sys.path.append('..')

import numpy as np

from bornagain.detector import edge_mask
from bornagain.viewers.qtviews import PADView
from bornagain.fileio.getters import CheetahFrameGetter
from bornagain.analysis import peaks
from bornagain.analysis import peaks_f

geom_file_name = 'data/cxin5016-oy-v1.geom'
cxi_file_name = 'data/cxilu5617-r0149-c00.cxi'

frame_getter = CheetahFrameGetter(cxi_file_name, geom_file_name)
dat = frame_getter.get_frame(0)
pads = dat['pad_data']
shape = pads[0].shape
masks = [edge_mask(d, 2) for d in pads]
# print(masks)
padview = PADView(frame_getter=frame_getter, mask_data=masks)

# pfind = peaks.PeakFinderV1(shape=masks[0].shape)
# padview.data_filters = pfind.snr_filter
#padview.data_filters = peaks.snr_filter_fortran

aft = np.asfortranarray

def peak_filter(self):
    for i in range(len(self.pad_data)):
        dat = self.pad_data[i].astype(np.float64)
        mask = self.mask_data[i].astype(np.float64) #np.ones_like(dat)
        # print(mask[0, 0], mask[10, 10])
        out = np.ones_like(dat)
        out = aft(out)
        nin = 3
        ncent = 6
        nout = 9
        peaks_f.peaker.boxsnr(aft(dat), aft(mask), out, nin, ncent, nout)
        self.pad_data[i] = out

padview.data_filters = [peak_filter]
#padview.add_roi()
padview.start()