import sys
import numpy as np
import matplotlib.pyplot as plt
from reborn import utils, source, detector
from reborn.viewers.qtviews import padviews
from reborn.viewers.mplviews.padviews import view_pad_data
from reborn.simulate.examples import LysozymeFrameGetter
def to_polar(vals, q, phi, n_q, q_range, n_phi, phi_range=None):
    q_size = (q_range[1] - q_range[0]) / float(n_q - 1)
    q_centers = np.linspace(q_range[0], q_range[1], n_q)
    q_edges = np.linspace(q_range[0] - q_size / 2, q_range[1] + q_size / 2, n_q + 1)
    q_min = q_edges[0]
    if phi_range is None:  # Then we go from 0 to 2pi...
        phi_size = 2*np.pi/n_phi
        phi_range = [phi_size/2, 2*np.pi-phi_size/2]
    else:
        phi_size = (phi_range[1] - phi_range[0]) / float(n_phi - 1)
    phi_centers = np.linspace(phi_range[0], phi_range[1], n_phi)
    phi_edges = np.linspace(phi_range[0] - phi_size / 2, phi_range[1] + phi_size / 2, n_phi + 1)
    phi_min = phi_edges[0]
    sum = np.zeros([n_q, n_phi])
    cnt = np.zeros([n_q, n_phi], dtype=int)
    for i in range(len(vals)):
        qi = q[i]
        pi = phi[i] % (2*np.pi)
        vi = vals[i]
        qind = int(np.floor((qi-q_min)/q_size))
        if qind >= n_q: continue
        if qind < 0: continue
        pind = int(np.floor((pi-phi_min)/phi_size))
        if pind >= n_phi: continue
        if pind < 0: continue
        cnt[qind, pind] += 1
        sum[qind, pind] += vi
    meen = np.divide(sum, cnt, out=np.zeros_like(sum), where=cnt!=0)
    return meen, cnt, q_centers, phi_centers

beam = source.Beam(photon_energy=9000*1.602e-19)
pads = detector.cspad_2x2_pad_geometry_list(detector_distance=0.1).binned(1)
q_mags = pads.q_mags(beam=beam)
phi = pads.azimuthal_angles(beam=beam)
fg = LysozymeFrameGetter(pad_geometry=pads, beam=beam)
# pv = padviews.PADView(frame_getter=fg)
# pv.start()
pat = fg.get_frame()
pat = pat['pad_data']
meen, cnt, qc, pc = to_polar(vals=detector.concat_pad_data(pat), q=q_mags, phi=phi, n_q=100,
                             q_range=[0, np.max(q_mags)], n_phi=100)
plt.figure(2)
plt.imshow(np.log10(meen+1), aspect='auto', interpolation='none',
           extent=[pc[0]/2/np.pi, pc[-1]/2/np.pi, qc[0]/1e10, qc[-1]/1e10])
plt.xlabel(r'$\phi/2\pi$')
plt.ylabel(r'$q$ [${\AA}^{-1}$]')
plt.figure(1)
view_pad_data(pad_geometry=pads, pad_data=[np.log10(p+1) for p in pat], show=False)
plt.show()
