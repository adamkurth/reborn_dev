import sys
sys.path.append('../..')

import numpy as np

from bornagain.detector import edge_mask
from bornagain.viewers.qtviews import PADView
from bornagain.fileio.getters import CheetahFrameGetter
from bornagain.analysis.peaks import boxsnr_numba, boxsnr_fortran, PeakFinder

geom_file_name = 'data/cxin5016-oy-v1.geom'
cxi_file_name = 'data/cxilu5617-r0149-c00.cxi'

frame_getter = CheetahFrameGetter(cxi_file_name, geom_file_name)
pad_geometry = frame_getter.pad_geometry
dat = frame_getter.get_frame(0)
pads = dat['pad_data']
shape = pads[0].shape
masks = [edge_mask(d, 1) for d in pads]
padview = PADView(pad_geometry=pad_geometry, frame_getter=frame_getter, mask_data=masks)

peak_finders = []
for i in range(len(masks)):
    peak_finders.append(PeakFinder(mask=masks[i], radii=(3, 6, 9)))

def peak_finder(self):
    centroids = [None]*self.n_pads
    n_peaks = 0
    for i in range(self.n_pads):
        dat = self.pad_data[i]
        mask = self.mask_data[i]
        pfind = peak_finders[i]
        pfind.find_peaks(data=dat, mask=mask)
        self.pad_data[i] = pfind.snr
        n_peaks += pfind.n_labels
        centroids[i] = pfind.centroids
    self.peaks = {'centroids': centroids, 'n_peaks': n_peaks}

padview.data_filters = [peak_finder]
padview.start()
