'''
Created on Sep 17, 2013

@author: kirian
'''

"""
Simple viewer
"""

# import sys
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import numpy as np
from pydiffract import convert, dataio, detector
import pydiffract.utils as utils





class myGLViewWidget(gl.GLViewWidget):

    def __init__(self):

        super(gl.GLViewWidget, self).__init__()
        self.opts = []

    def pan(self, dx, dy, dz, relative=False):
        """
        Moves the center (look-at) position while holding the camera in place. 
        
        If relative=True, then the coordinates are interpreted such that x
        if in the global xy plane and points to the right side of the view, y is
        in the global xy plane and orthogonal to x, and z points in the global z
        direction. Distances are scaled roughly such that a value of 1.0 moves
        by one pixel on screen.
        
        """
        if not relative:
            self.opts['center'] += QtGui.QVector3D(dx, dy, dz)
        else:
            cPos = self.cameraPosition()
            cVec = self.opts['center'] - cPos
            dist = cVec.length()  # # distance from camera to center
            xDist = dist * 2. * np.tan(0.5 * self.opts['fov'] * np.pi / 180.)  # # approx. width of view at distance of center point
            xScale = xDist / self.width()
            zVec = QtGui.QVector3D(0, 0, 1)
            xVec = QtGui.QVector3D.crossProduct(zVec, cVec).normalized()
            # xVec = QtGui.QVector3D(0,1,0)
            # yVec = QtGui.QVector3D.crossProduct(xVec, zVec).normalized()
            yVec = QtGui.QVector3D(0, 0, 1)
            self.opts['center'] = self.opts['center'] + xVec * xScale * dx + yVec * xScale * dy + zVec * xScale * dz
        self.update()







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

        menubar = self.menuBar()

        fileMenu = menubar.addMenu('&File')

        exitAction = QtGui.QAction(QtGui.QIcon('exit.png'), '&Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(QtGui.qApp.quit)
        fileMenu.addAction(exitAction)

        openFile = QtGui.QAction(QtGui.QIcon('open.png'), '&Open', self)
        openFile.setShortcut('Ctrl+O')
        openFile.setStatusTip('Open image file')
        openFile.triggered.connect(self.showOpenFileDialog)
        fileMenu.addAction(openFile)

        viewMenu = menubar.addMenu('View')

        showScan = QtGui.QAction(QtGui.QIcon('open.png'), 'Fast scan', self)
        showScan.setStatusTip('Show fast-scan directions')
        showScan.triggered.connect(self.showFastScans)
        viewMenu.addAction(showScan)

        framePanels = QtGui.QAction(QtGui.QIcon('open.png'), 'Panel frames', self)
        framePanels.setStatusTip('Draw frames around panels')
        framePanels.triggered.connect(self.framePanels)
        viewMenu.addAction(framePanels)

        showAxes = QtGui.QAction(QtGui.QIcon('open.png'), 'Axes', self)
        showAxes.setStatusTip('Draw axes (red,green,blue = x,y,z)')
        showAxes.triggered.connect(self.showAxes)
        viewMenu.addAction(showAxes)

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

    def framePanels(self):

        col = (0, 0, 1, 0.5)

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

    def showAxes(self):

        axislen = self.panelList.pixSize.copy() * 100
        pos = np.zeros((2, 3))
        pos[0] = np.array([0, 0, 0])
        pos[1] = np.array([axislen, 0, 0])
        col = (1, 0, 0, 1.0)
        beamd = gl.GLLinePlotItem(pos=pos, color=col)
        self.w.addItem(beamd)

        pos = np.zeros((2, 3))
        pos[0] = np.array([0, 0, 0])
        pos[1] = np.array([0, axislen, 0])
        col = (0, 1, 0, 1.0)
        beamd = gl.GLLinePlotItem(pos=pos, color=col)
        self.w.addItem(beamd)

        pos = np.zeros((2, 3))
        pos[0] = np.array([0, 0, 0])
        pos[1] = np.array([0, 0, axislen])
        col = (0, 0, 1, 1.0)
        beamd = gl.GLLinePlotItem(pos=pos, color=col)
        self.w.addItem(beamd)

    def showBeamDirection(self):

        pos = np.zeros((2, 3))
        pos[0] = np.array([0, 0, 0])
        pos[1] = np.array([0, 0, self.panelList.pixSize * 500])
        col = (0, 0, 1, 1.0)
        beamd = gl.GLLinePlotItem(pos=pos, color=col)
        self.w.addItem(beamd)

    def showInteractionPoint(self):

        # Add point at origin
        pos = np.empty((1, 3))
        pos[0] = (0, 0, 0)
        col = (0, 1, 0, 0.4)
        siz = p.pixSize * 10
        sp = gl.GLScatterPlotItem(pos=pos, size=siz, color=col, pxMode=False)
        w.addItem(sp)

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

#     [pa, reader] = convert.crystfelToPanelList("examples/example1.geom")
#     reader.getShot(pa, "examples/example1.h5")

    filePath = '../examples/sacla/02-2014/run178730.hdf5'

    reader = dataio.saclaReader(filePath)
    pa = detector.panelList()
    reader.getFrame(pa, 0)
    for p in pa:
        p.T = p.T + np.array([0, 0, 0.05])

    g.panelList = pa

    c = pa.getCenter()

    w.pan(c[0], c[1], c[2])
    w.setCameraPosition(distance=c[2] * 2)
#     w.orbit(45, 60)
    w.orbit(45, -120)

    ims = []
    cs = []
    i = -1

    mn = -100
    padata = pa.data
    padata[padata < mn] = mn
    pa.data = padata
    maxd = np.max(padata)
    mind = np.min(padata)

    for p in pa:

        i += 1

        g.showPanel(p, rang=[mind, maxd])

        # Add points at centers of panels
#         pos = np.empty((1, 3))
#         pos[0] = p.getCenter()
#         col = (0, 1, 0, 0.4)
#         siz = p.pixSize.copy() * 10
#         sp = gl.GLScatterPlotItem(pos=pos, size=siz, color=col, pxMode=False)
#         w.addItem(sp)

    w.renderText(0, 0, -0.05, 'test')

    g.showPanels()

#     im = gl.GLImageItem(pg.makeRGBA(d)[0])
#     im.scale(pix, pix, pix)
#     w.addItem(im)

#     # Indicate fast-scan direction
#     pos = np.zeros((2, 3))
#     pos[0] = (0, 0, 0)
#     pos[1] = (1, 0, 0) * 10  # p.pixSize * 1000
#     col = (1, 0, 0, 1.0)
#     fs = gl.GLLinePlotItem(pos=pos, color=col)
#     w.addItem(fs)

    app.exec_()
#     sys.exit(app.exec_())

if __name__ == '__main__':
    main()
