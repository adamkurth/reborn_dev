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

class gui(QtGui.QMainWindow):

    def __init__(self):

        super(gui, self).__init__()
        self.initUI()
        self.w = gl.GLViewWidget()
        self.setCentralWidget(self.w)
        self.setWindowTitle('gwiz')
        self.setWindowState(self.windowState() & ~QtCore.Qt.WindowMinimized | QtCore.Qt.WindowActive)
        self.activateWindow()
        self.show()

    def initUI(self):

        self.resize(1000, 1000)
        self.statusBar()
        self.center()

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

def getPanelRotation(p):

    """ Return the rotation angle, axis, and matrix for orienting panel in gl view."""

    V1 = np.zeros((2, 3))
    V1[0, :] = p.F
    V1[1, :] = p.S
    V2 = np.zeros((2, 3))
    V2[0, 0] = 1
    V2[1, 1] = 1
    R = utils.kabschRotation(V2, V1)
    V, phi = utils.axisAndAngle(R)
    V = V.T

    return phi, V, R

def main():

    app = QtGui.QApplication([])
    g = gui()
    w = g.w

    [pa, reader] = convert.crystfelToPanelList("examples/example1.geom")
    reader.getShot(pa, "examples/example1.h5")

    w.setCameraPosition(distance=0.5 * pa[0].T[2] / pa[0].pixSize)
    w.orbit(45, -120)

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

        d[0, :] = maxd * 0.5
        d[0:10, 0] = maxd * 0.5

        d = d.copy().T

        d -= mind
        d /= maxd
        d *= 2 ** 8

        dd = pg.makeRGBA(d)[0]

        ims.append(gl.GLImageItem(dd))
        im = ims[i]

        phi, V, R = getPanelRotation(p)
        T = p.T / p.pixSize
        phi *= 180 / np.pi

        im.translate(-0.5, -0.5, 0)
        im.rotate(phi, V[0], V[1], V[2])
        im.translate(T[0], T[1], T[2])

        i += 1

        w.addItem(im)

    # Scatter plot items

    npeak = 2
    pos = np.empty((npeak, 3))
    size = np.empty((npeak))
    color = np.empty((npeak, 4))
    pos[0] = (100, 0, 0); size[0] = 10;   color[0] = (0.0, 1.0, 0.0, 0.4)
    pos[1] = (0, 100, 0); size[1] = 10;   color[1] = (0.0, 1.0, 0.0, 0.4)

    sp1 = gl.GLScatterPlotItem(pos=pos, size=size, color=color, pxMode=False)
    sp1.rotate(phi, V[0], V[1], V[2])
    sp1.translate(T[0], T[1], T[2])
    w.addItem(sp1)

    # Line plot items

    for p in pa:

        T = p.T / p.pixSize
        phi, V, R = getPanelRotation(p)
        phi *= 180 / np.pi

        fs = np.array([[0, 0, 0], p.F]) / p.pixSize * p.nF * 0.3

        plt = gl.GLLinePlotItem(pos=fs, color=pg.glColor((0, 1 * 1.3)))
#         plt.translate(-0.5, -0.5, 0)
        plt.rotate(phi, V[0], V[1], V[2])
        plt.translate(T[0], T[1], T[2])
        w.addItem(plt)

        b = (p.getVertices(loop=True) + p.T) / p.pixSize

        plt2 = gl.GLLinePlotItem(pos=b, color=pg.glColor((10, 30 * 1.3)))
        w.addItem(plt2)



    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
