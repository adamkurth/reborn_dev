'''
Created on Aug 4, 2013

@author: kirian
'''

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import numpy as np
import pydiffract.convert as convert
import pydiffract.utils as utils

[pa, reader] = convert.crystfelToPanelList("examples/example1.geom")
reader.getShot(pa, "examples/example1.h5")

# # Create a GL View widget to display data
app = QtGui.QApplication([])
w = gl.GLViewWidget()
w.resize(1000, 1000)
w.show()
w.setWindowTitle('gwiz')
w.setCameraPosition(distance=0.5 * pa[0].T[2] / pa[0].pixSize)
w.orbit(45, -120)

# p = pa[0]
# p.T[0] = p.pixSize * 1
# p.T[1] = p.pixSize * 2
# p.T[2] = 0  # p.pixSize * 3
# p.F = np.array([1, 0, 0]) * p.pixSize
# p.S = np.array([0, 1, 0]) * p.pixSize
# p.data[0, :] = 200

ims = []
i = 0

padata = pa.data
padata[padata < 0] = 0
pa.data = padata
# padata = pa.solidAngle
maxd = np.max(padata)
mind = np.min(padata)

# maxd = 1e-5

for p in pa:

    d = p.data
#     d = p.solidAngle.copy().reshape(p.nS, p.nF)

    d[0, :] = maxd * 0.5
    d[0:10, 0] = maxd * 0.5

    d = d.copy().T

    d -= mind
    d /= maxd
    d *= 2 ** 8

    dd = pg.makeRGBA(d)[0]

    ims.append(gl.GLImageItem(dd))
    im = ims[i]

    V1 = np.zeros((2, 3))
    V1[0, :] = p.F
    V1[1, :] = p.S

    V2 = np.zeros((2, 3))
    V2[0, 0] = 1
    V2[1, 1] = 1
    R = utils.kabschRotation(V2, V1)
#     R = R.T
    V, phi = utils.axisAndAngle(R)
    V = V.T

    T = (p.T) / p.pixSize

    T[0] += 0.2 * T[0]

    phi *= 180 / np.pi

    print(p.F)
    print(p.S)
    print(T)
    print(R)
    print(V)
    print(phi)

    im.translate(-0.5, -0.5, 0)
    im.rotate(phi, V[0], V[1], V[2])
    im.translate(T[0], T[1], T[2])

    i += 1

    w.addItem(im)

#     break

ax = gl.GLAxisItem()
w.addItem(ax)

# xgrid = gl.GLGridItem()
# ygrid = gl.GLGridItem()
# zgrid = gl.GLGridItem()
# xgrid.rotate(90, 0, 1, 0)
# ygrid.rotate(90, 1, 0, 0)
# w.addItem(xgrid)
# w.addItem(ygrid)
# w.addItem(zgrid)

# Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
