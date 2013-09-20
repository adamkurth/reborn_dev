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

        d[d < mn] = mn
        d[d > mx] = mx
        d -= mn
        d /= mx
        d *= 255

        # Geometry data
        phi, V, R = getPanelRotation(p)
#         junk = getPanelRotation2(p)
        T = p.T
        phi = np.degrees(phi)
        pix = p.pixSize

        # Add intensity data to view
        im = gl.GLImageItem(pg.makeRGBA(d.T)[0])
        im.scale(pix, pix, pix)
        im.translate(-pix / 2, -pix / 2, 0)
        im.rotate(phi, V[0], V[1], V[2])
        im.translate(T[0], T[1], T[2])
        self.w.addItem(im)


def getPanelRotation(p):

    R = np.zeros([3, 3])

    R[:, 0] = utils.vecNorm(p.F[np.newaxis])
    R[:, 1] = utils.vecNorm(p.S[np.newaxis])
    R[:, 2] = utils.vecNorm((np.cross(R[:, 0], R[:, 1]))[np.newaxis])

    V, phi = utils.axisAndAngle(R)
    V = V.T

    return phi, V, R

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

#     # Indicate fast-scan direction
#     pos = np.zeros((2, 3))
#     pos[0] = (0, 0, 0)
#     pos[1] = (1, 0, 0) * 10  # p.pixSize * 1000
#     col = (1, 0, 0, 1.0)
#     fs = gl.GLLinePlotItem(pos=pos, color=col)
#     w.addItem(fs)

    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
