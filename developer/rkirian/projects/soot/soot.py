""" Example of how to generate a polar-binned diffraction intensity map. """
import numpy as np
import matplotlib.pyplot as plt
from reborn import source, detector
from reborn.viewers.mplviews.padviews import view_pad_data
from reborn.simulate.examples import LysozymeFrameGetter
beam = source.Beam(photon_energy=9000*1.602e-19)
pads = detector.cspad_2x2_pad_geometry_list(detector_distance=0.1).binned(1)
polar_assembler = detector.PolarPADAssembler(pad_geometry=pads, beam=beam, n_q_bins=100, n_phi_bins=100)
fg = LysozymeFrameGetter(pad_geometry=pads, beam=beam)
fg.view()
pat = fg.get_frame()
data = pat['pad_data']
meen = polar_assembler.get_mean(data)
pc = polar_assembler.phi_bin_centers
qc = polar_assembler.q_bin_centers
plt.imshow(np.log10(meen+1), aspect='auto', interpolation='none',
           extent=[pc[0]/2/np.pi, pc[-1]/2/np.pi, qc[0]/1e10, qc[-1]/1e10])
plt.xlabel(r'$\phi/2\pi$')
plt.ylabel(r'$q$ [${\AA}^{-1}$]')
view_pad_data(pad_geometry=pads, data=[np.log10(p + 1) for p in data], show=False)
plt.show()
