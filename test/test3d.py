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

import convert

# # Create a GL View widget to display data
app = QtGui.QApplication([])
w = gl.GLViewWidget()
w.show()
w.setWindowTitle('gwiz')
w.setCameraPosition(distance=1000)

pa = convert.crystfel_to_panel_list("examples/example1.geom")

pa.read("examples/example1.h5")

pa.computeRealSpaceGeometry()

p = pa[0]
x = p.V[:, 0].reshape([p.nS, p.nF])
y = p.V[:, 1].reshape([p.nS, p.nF])
z = p.V[:, 2].reshape([p.nS, p.nF])
surf = gl.GLSurfacePlotItem(x=x, y=y, z=z, shader='normalColor')
w.addItem(surf)


# # Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
