'''
Created on Sep 17, 2013

@author: kirian
'''

"""
Simple viewer
"""

import sys
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import numpy as np
import pydiffract.convert as convert
import pydiffract.utils as utils

class panelListGui(QtGui.QMainWindow):

    def __init__(self):

        super(panelListGui, self).__init__()
        self.initUI()
        self.w = gl.GLViewWidget()
        self.setCentralWidget(self.w)
        self.setWindowTitle('gwiz')
        self.setWindowState(self.windowState() & ~QtCore.Qt.WindowMinimized | QtCore.Qt.WindowActive)
        self.activateWindow()
        self.show()
        self.panelList = None

    def initUI(self):

        self.resize(1000, 1000)
        self.statusBar()
#         self.center()

        exitAction = QtGui.QAction(QtGui.QIcon('exit.png'), '&Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(QtGui.qApp.quit)

        openFile = QtGui.QAction(QtGui.QIcon('open.png'), '&Open', self)
        openFile.setShortcut('Ctrl+O')
        openFile.setStatusTip('Open image file')
        openFile.triggered.connect(self.showOpenFileDialog)

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(exitAction)
        fileMenu.addAction(openFile)

    def center(self):

        qr = self.frameGeometry()
        cp = QtGui.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def loadImage(self, pl):

        pass

    def showOpenFileDialog(self):

        fname = QtGui.QFileDialog.getOpenFileName(self, 'Open file', '$HOME')
        self.loadImage(fname)

    def showPanels(self):

        for p in self.panelList:

            self.showPanel(p)

    def framePanels(self, col=(0, 0, 1, 0.5)):

        for p in self.panelList:
            pos = np.zeros((5, 3))
            pos[:] = p.getVertices(edge=True, loop=True)
            fr = gl.GLLinePlotItem(pos=pos, color=col)
            self.w.addItem(fr)

    def showFastScans(self):

        for p in self.panelList:
            pos = np.zeros((2, 3))
            pos[0] = p.T
            pos[1] = p.T + p.F * p.nF * 0.5
            col = (1, 0, 0, 1.0)
            fs = gl.GLLinePlotItem(pos=pos, color=col)
            self.w.addItem(fs)

    def showPanel(self, p, rang=None):

        # Intensity data
        d = p.data.copy()

        if rang is None:
            rang = np.array([d.min(), d.max()])

        mn = rang[0]
        mx = rang[1]

        if rang is None:
            rang = np.array([d.min(), d.max()])

        mn = rang[0]
        mx = rang[1]

        hi = mx * 0.5
        d[0:10, :] = hi
        d[0:20, 0:20] = hi
        d[-5:-1, :] = hi
        d[:, -5:-1] = hi
        d[d < mn] = mn
        d[d > mx] = mx
        d -= mn
        d /= mx
        d *= 255

        # Geometry data
        phi, V, R = getPanelRotation(p)
        T = p.T
        phi = np.degrees(phi)
        pix = p.pixSize

        # Add intensity data to view
        im = gl.GLImageItem(pg.makeRGBA(d)[0])
        im.scale(pix, pix, pix)
        im.translate(-pix / 2, -pix / 2, 0)
        im.rotate(phi, V[0], V[1], V[2])
        im.translate(T[0], T[1], T[2])
        self.w.addItem(im)


def getPanelRotation(p):

    """ Return the rotation angle, axis, and matrix for orienting panel in gl view."""

    V1 = np.zeros((2, 3))
    V1[0, :] = p.F
    V1[1, :] = p.S
    V1 = utils.vecNorm(V1)
    V2 = np.zeros((2, 3))
    V2[0, 0] = 1
    V2[1, 1] = 1
    R = utils.kabschRotation(V1, V2)
#     print(V1)
#     print(V2)
#     print(R)
    err = np.max(V1 - (R.dot(V2.T)).T)
    print(np.max(V1 - (R.dot(V2.T)).T))
    V, phi = utils.axisAndAngle(R)
    V = V.T

    return phi, V, R


def R_axis_angle(matrix, axis, angle):
    """Generate the rotation matrix from the axis-angle notation.

    Conversion equations
    ====================

    From Wikipedia (http://en.wikipedia.org/wiki/Rotation_matrix), the conversion is given by::

        c = cos(angle); s = sin(angle); C = 1-c
        xs = x*s;   ys = y*s;   zs = z*s
        xC = x*C;   yC = y*C;   zC = z*C
        xyC = x*yC; yzC = y*zC; zxC = z*xC
        [ x*xC+c   xyC-zs   zxC+ys ]
        [ xyC+zs   y*yC+c   yzC-xs ]
        [ zxC-ys   yzC+xs   z*zC+c ]


    @param matrix:  The 3x3 rotation matrix to update.
    @type matrix:   3x3 numpy array
    @param axis:    The 3D rotation axis.
    @type axis:     numpy array, len 3
    @param angle:   The rotation angle.
    @type angle:    float
    """

    # Trig factors.
    ca = cos(angle)
    sa = sin(angle)
    C = 1 - ca

    # Depack the axis.
    x, y, z = axis

    # Multiplications (to remove duplicate calculations).
    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    # Update the rotation matrix.
    matrix[0, 0] = x * xC + ca
    matrix[0, 1] = xyC - zs
    matrix[0, 2] = zxC + ys
    matrix[1, 0] = xyC + zs
    matrix[1, 1] = y * yC + ca
    matrix[1, 2] = yzC - xs
    matrix[2, 0] = zxC - ys
    matrix[2, 1] = yzC + xs
    matrix[2, 2] = z * zC + ca


def main():

    app = QtGui.QApplication([])
    g = panelListGui()
    w = g.w

    [pa, reader] = convert.crystfelToPanelList("examples/example1.geom")
    reader.getShot(pa, "examples/example1.h5")

    g.panelList = pa

    c = pa.getCenter()
    w.setCameraPosition(distance=c[2] * 3)
    w.orbit(45, 60)

    ims = []
    cs = []
    i = -1

    mn = -100
    padata = pa.data
    padata[padata < mn] = mn
    pa.data = padata
    # padata = pa.solidAngle
    maxd = np.max(padata)
    mind = np.min(padata)

    # maxd = 1e-5

    for p in pa:

        i += 1

        g.showPanel(p, rang=[mind, maxd])

        # Add points at centers of panels
        pos = np.empty((1, 3))
        pos[0] = p.getCenter()
        col = (0, 1, 0, 0.4)
        siz = p.pixSize * 10
        sp = gl.GLScatterPlotItem(pos=pos, size=siz, color=col, pxMode=False)
        w.addItem(sp)

    g.showPanels()
    g.framePanels()
    g.showFastScans()

#     im = gl.GLImageItem(pg.makeRGBA(d)[0])
#     im.scale(pix, pix, pix)
#     w.addItem(im)

    # Add point at origin
    pos = np.empty((1, 3))
    pos[0] = (0, 0, 0)
    col = (0, 1, 0, 0.4)
    siz = p.pixSize * 10
    sp = gl.GLScatterPlotItem(pos=pos, size=siz, color=col, pxMode=False)
    w.addItem(sp)

    # Indicate fast-scan direction
    pos = np.zeros((2, 3))
    pos[0] = (0, 0, 0)
    pos[1] = (1, 0, 0) * 10  # p.pixSize * 1000
    col = (1, 0, 0, 1.0)
    fs = gl.GLLinePlotItem(pos=pos, color=col)
    w.addItem(fs)


    # Line plot items

#     for p in pa:
#
#
#
#         T = p.T / p.pixSize
#         phi, V, R = getPanelRotation(p)
#         phi *= 180 / np.pi
#
#         fs = np.array([[0, 0, 0], p.F]) / p.pixSize * p.nF * 0.3
#
#         plt = gl.GLLinePlotItem(pos=fs, color=pg.glColor((0, 1 * 1.3)))
# #         plt.translate(-0.5, -0.5, 0)
#         plt.rotate(phi, V[0], V[1], V[2])
#         plt.translate(T[0], T[1], T[2])
#         w.addItem(plt)
#
#         b = (p.getVertices(loop=True) + p.T) / p.pixSize
#
#         plt2 = gl.GLLinePlotItem(pos=b, color=pg.glColor((10, 30 * 1.3)))
#         w.addItem(plt2)



    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
