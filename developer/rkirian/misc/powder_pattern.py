import h5py
import numpy as np
import pyqtgraph as pg
from reborn.detector import RadialProfiler
from reborn.source import Beam
from reborn.external import crystfel
from reborn.viewers.qtviews.padviews import PADView

import scipy.constants as const

keV = const.value('electron volt')*1000

pads = crystfel.geometry_file_to_pad_geometry_list('../data/cxilu5617-taw10.geom')
for p in pads:
    p.t_vec[2] = 0.1156
geom = crystfel.load_crystfel_geometry('../data/cxilu5617-taw10.geom')
beam = Beam(photon_energy=9.6/keV)

f = h5py.File('./data/r0230-detector0-class1-sum.h5')
dat = np.array(f['/data/data'])
f.close()
dats = crystfel.split_image(dat, geom)

f = h5py.File('./data/f1_mask.h5')
mask1 = np.ravel(crystfel.split_image(np.array(f['/data/data']), geom))
f.close()

f = h5py.File('./data/f2_mask.h5')
mask2 = np.ravel(crystfel.split_image(np.array(f['/data/data']), geom))
f.close()

qmags = np.ravel([p.q_mags(beam=beam) for p in pads])

d = 1e10*2*np.pi/qmags
print(np.min(d), np.max(d))

radial = RadialProfiler()
radial.make_plan(q_mags=qmags, n_bins=500, mask=mask1)
powder1 = radial.get_profile(np.ravel([d for d in dats]), average=True)
bins1 = radial.bin_centers.copy()
radial = RadialProfiler()
radial.make_plan(q_mags=qmags, n_bins=500, mask=mask2)
powder2 = radial.get_profile(np.ravel([d for d in dats]), average=True)
bins2 = radial.bin_centers.copy()

plot = pg.plot(bins1, powder1, pen='r')
plot.plot(bins2, powder2*(np.sum(powder1)/np.sum(powder2)), pen='b', line=None)

showme = [p.reshape(p.q_mags(beam=beam))*1e10 for p in pads]

# print(showme)

padview = PADView(raw_data=showme, pad_geometry=pads)
padview.start()
