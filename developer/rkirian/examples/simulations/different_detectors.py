import numpy as np
from reborn import source, detector
from reborn.simulate.solutions import get_pad_solution_intensity
from reborn.viewers.qtviews import PADView
beam = source.Beam(photon_energy=9e3*1.602e-19, pulse_energy=1e-3)
pads1 = detector.tiled_pad_geometry_list(pixel_size=8.9e-5, pad_shape=(3840, 3840), tiling_shape=(1, 1), distance=1.5)
pads2 = detector.epix10k_pad_geometry_list(detector_distance=0)
theta = np.pi/4
R = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
for p in pads2:
    p.fs_vec = np.dot(p.fs_vec, R.T)
    p.ss_vec = np.dot(p.ss_vec, R.T)
    p.t_vec = np.dot(p.t_vec, R.T)
    p.t_vec[0] += 100e-3
    p.t_vec[2] += 0.2
pads = detector.PADGeometryList(pads1 + pads2)
intsty = get_pad_solution_intensity(beam=beam, pad_geometry=pads, thickness=60e-6, liquid='water', poisson=False)
intsty = pads.concat_data(intsty)
intsty /= pads.solid_angles()
pv = PADView(pad_geometry=pads, raw_data=intsty)
pv.start()