'''
Created on Aug 4, 2013

@author: kirian
'''

# -*- coding: utf-8 -*-
"""
This example demonstrates the use of GLSurfacePlotItem.
"""


# # Add path to library (just for examples; you do not need this)
# import initExample

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
w.show()
w.setWindowTitle('gwiz')
w.setCameraPosition(distance=1000)



ims = []
i = 0

for p in pa:

    d = p.solidAngle.copy().reshape(p.nS, p.nF)
#     dd = np.empty((d.shape[0], d.shape[1], 3))

#     dd[:, :, 0] = d
#     dd[:, :, 1] = d
#     dd[:, :, 2] = d

#     dd -= np.min(dd)
#     dd /= np.max(dd)
#     dd *= 2 ** 8
#     dd = np.ubyte(dd)

    dd = pg.makeRGBA(d)[0]

    ims.append(gl.GLImageItem(dd))
    im = ims[i]
    w.addItem(im)

    V1 = np.zeros((2, 3))
    V1[0, :] = p.F
    V1[1, :] = p.S

    V2 = np.zeros((2, 3))
    V2[0, 0] = 1
    V2[1, 1] = 1
    R = utils.kabschRotation(V2, V1)
    V, phi = utils.axisAndAngle(R.T)
    V = V.T

    T = (p.T) / p.pixSize
    Tc = (p.F + p.S) / (2.0 * p.pixSize)
    Tcc = T + Tc

    im.translate(-Tc[0], -Tc[1], -Tc[2])
    im.rotate(V[0], V[1], V[2], phi)
    im.translate(Tcc[0], Tcc[1], Tcc[2])

    i += 1





xgrid = gl.GLGridItem()
ygrid = gl.GLGridItem()
zgrid = gl.GLGridItem()
# w.addItem(xgrid)
# w.addItem(ygrid)
w.addItem(zgrid)

# # rotate x and y grids to face the correct direction
# xgrid.rotate(90, 0, 1, 0)
# ygrid.rotate(90, 1, 0, 0)

# # Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
