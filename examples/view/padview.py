import sys
sys.path.append('../..')

from bornagain.simulate import examples
from bornagain.viewers.qtviews import PADView
from bornagain import detector
import numpy as np
# import pyqtgraph as pg


pads = []

pad = detector.PADGeometry()
pad.simple_setup(pixel_size=100e-6, distance=0.1)
pads.append(pad)

pad = detector.PADGeometry()
pad.simple_setup(pixel_size=100e-6, distance=0.1)
ang = 20*np.pi/180.
R = np.array([[np.cos(ang), np.sin(ang), 0],[-np.sin(ang), np.cos(ang), 0],[0, 0, 1]])
f = pad.fs_vec.copy()
s = pad.ss_vec.copy()
pad.fs_vec = np.dot(f, R)
pad.ss_vec = np.dot(s, R)
t = pad.t_vec.copy().ravel()
t[0] += 0.04
t[1] += 0.08
pad.t_vec = t
pads.append(pad)

if True:
    pad = detector.PADGeometry()
    pad.simple_setup(pixel_size=100e-6, distance=0.1)
    ang = 30 * np.pi / 180.
    R = np.array([[np.cos(ang), np.sin(ang), 0], [-np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
    f = pad.fs_vec.copy()
    s = pad.ss_vec.copy()
    f = np.dot(f, R).ravel()
    s = np.dot(s, R).ravel()
    f[1] *= -1
    s[1] *= -1
    pad.fs_vec = f
    pad.ss_vec = s
    t = pad.t_vec.copy().ravel()
    t[0] += 0.04
    t[1] += 0.08
    t[1] *= -1
    pad.t_vec = t
    pads.append(pad)

pads = examples.cspad_pads()
# pads = examples.pnccd_pads()

sim = examples.lysozyme_molecule(pads=pads)

geom = sim['pad_geometry']
I = sim['intensity']
# I = [np.random.poisson(d*1e-5) + 1 for d in I]
# I = [np.log10(d) for d in I]
padgui = PADView(pad_data=I, pad_geometry=geom, logscale=True)

# padgui.show_coordinate_axes()
# padgui.add_grid()
# padgui.show_pad_labels()
# padgui.add_rings([200, 400, 600, 800], pens=[pg.mkPen([255, 0, 0], width=2)]*4)

padgui.start()