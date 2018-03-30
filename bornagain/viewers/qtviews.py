import numpy                 as np
import pyqtgraph             as pg
import pyqtgraph.opengl      as gl
from   pyqtgraph.Qt import QtCore, QtGui

"""
This is supposed to have various viewers that use pyqtgraph.  It's mostly useless right now.
"""

# Some default bright colors.  Might need to make this list longer in the future.
colors = [pg.glColor([255, 0, 0]),
          pg.glColor([0, 255, 0]),
          pg.glColor([0, 0, 255]),
          pg.glColor([255, 255, 0]),
          pg.glColor([0, 255, 255]),
          pg.glColor([255, 0, 255]),
          pg.glColor([255, 255, 255]),
          pg.glColor([255, 128, 128])]

def bright_colors(i):

    """ Some nice colors.  Only 8 available, which loops around as the input index increments."""

    return colors[i % len(colors)]

class Volumetric3D(object):
    ''' View a 3D density map '''

    def __init__(self):

        self.app = QtGui.QApplication([])
        self.w = gl.GLViewWidget()
        self.maxDist = 0
        self.defaultWidth = 5
        self.dat = None
        self.datSum = None
        self.defaultColor = pg.glColor([255, 255, 255])

    def add_density(self, rho, color=None):

        if color is None:
            col = [255, 255, 255]
        else:
            col = color

        if self.dat is None:
            self.dat = np.zeros(rho.shape + (4,))
            self.datSum = np.zeros(rho.shape)

        # This is needed for scaling the view -- need ot know size of data
        self.maxDist = max(self.maxDist, max(self.datSum.shape) / np.sqrt(2))

        self.datSum += rho
        self.dat[..., 0] += rho * col[0]
        self.dat[..., 1] += rho * col[1]
        self.dat[..., 2] += rho * col[2]
        self.dat[..., 3] += rho

    def add_grid(self):

        g = gl.GLGridItem()
        g.scale(10, 10, 1)
        self.w.addItem(g)

    def add_lines(self, r, color=None, width=None):

        if color is None:
            col = self.defaultColor
        else:
            col = pg.glColor(color)

        if width is None:
            wid = self.defaultWidth
        else:
            wid = width

        plt = gl.GLLinePlotItem(pos=r, mode='lines', width=wid, color=col)
        self.w.addItem(plt)

    def add_rgb_axis(self, length=None, width=None):

        if width is None:
            wid = self.defaultWidth
        else:
            wid = width

        if length is None:
            axlen = self.maxDist
        else:
            axlen = length

        self.add_lines(np.array([[0, 0, 0], [1, 0, 0]]) * axlen, [255, 0, 0], width=wid)
        self.add_lines(np.array([[0, 0, 0], [0, 1, 0]]) * axlen, [0, 255, 0], width=wid)
        self.add_lines(np.array([[0, 0, 0], [0, 0, 1]]) * axlen, [0, 0, 255], width=wid)

    def show(self, smooth=True):

        self.dat[..., 0:-1] *= 255. / self.dat[..., 0:-1].max()
        self.dat[..., 3] *= 255. / self.dat[..., 3].max()
        self.dat = np.ubyte(self.dat)
        v = gl.GLVolumeItem(self.dat, smooth=smooth)
        v.translate(-self.dat.shape[0] / 2., -self.dat.shape[1] / 2., -self.dat.shape[2] / 2.)

        self.w.addItem(v)
        self.w.setCameraPosition(distance=self.maxDist * 2)
        self.w.show()

        QtGui.QApplication.instance().exec_()


def MapProjection(data, axis=None):
    ''' View a 3D density map as a projection along selected axes (which can be a list)'''

    if axis is not None:
        if type(axis) is not list:
            axis = [axis]
        dat = []
        for ax in axis:
            dat.append(np.sum(data, axis=ax))
        dat = np.concatenate(dat)
    else:
        dat = data
    app = QtGui.QApplication([])
    win = QtGui.QMainWindow()
    win.resize(800, 800)
    imv = pg.ImageView()
    win.setCentralWidget(imv)
    win.show()
    # 	win.setWindowTitle('pyqtgraph example: ImageView')
    imv.setImage(dat, xvals=np.linspace(1., 3., dat.shape[0]))
    QtGui.QApplication.instance().exec_()


def MapSlices(data, axis=None, levels=None):
    ''' View a 3D density map as a projection along selected axes (which can be a list)'''

    # This assumes we have a cube...
    n = data.shape[0]
    m = int((data.shape[0] - 1) / 2)

    if axis is None:
        axis = [0, 1, 2]

    if axis is not None:
        if type(axis) is not list:
            axis = [axis]
        dat = []
        for ax in axis:
            if ax == 0:
                dat.append(np.reshape(data[m, :, :], [n, n]))
            elif ax == 1:
                dat.append(np.reshape(data[:, m, :], [n, n]))
            elif ax == 2:
                dat.append(np.reshape(data[:, :, m], [n, n]))
        dat = np.concatenate(dat)
    else:
        dat = data
    app = QtGui.QApplication([])
    win = QtGui.QMainWindow()
    win.resize(800, 800)
    imv = pg.ImageView()
    win.setCentralWidget(imv)
    win.show()
    # 	win.setWindowTitle('pyqtgraph example: ImageView')
    if levels is not None:
        autoLevels = False
    else:
        autoLevels = True
    imv.setImage(dat, xvals=np.linspace(1., 3., dat.shape[0]), levels=levels, autoLevels=autoLevels)
    QtGui.QApplication.instance().exec_()


class Scatter3D(object):

    ''' Simple viewer for 3D scatter plots. '''

    def __init__(self):

        self.app = pg.QtGui.QApplication([])
        self.w = gl.GLViewWidget()
        self.defaultColor = pg.glColor([255, 255, 255])
        self.defaultSize = 1
        self.defaultWidth = 1
        self.maxDist = 0

    def add_points(self, r, color=None, size=None):

        ''' Add an Nx3 array of points r with specified color and size.  Color is a 3-element
		    Python list and size is a float scalar. '''

        if color is None:
            col = self.defaultColor
        else:
            col = pg.glColor(color)

        if size is None:
            siz = self.defaultSize
        else:
            siz = size

        self.maxDist = max(self.maxDist, np.amax(np.sqrt(np.sum(r * r, axis=1))))
        plt = gl.GLScatterPlotItem(pos=r, color=col, size=siz)
        self.w.addItem(plt)

    def add_lines(self, r, color=None, width=None):

        if color is None:
            col = self.defaultColor
        else:
            col = pg.glColor(color)

        if width is None:
            wid = self.defaultWidth
        else:
            wid = width

        plt = gl.GLLinePlotItem(pos=r, mode='lines', width=wid, color=col)
        self.w.addItem(plt)

    def add_rgb_axis(self, length=None, width=None):

        if width is None:
            wid = self.defaultWidth
        else:
            wid = width

        if length is None:
            axlen = self.maxDist
        else:
            axlen = length

        self.add_lines(np.array([[0, 0, 0], [1, 0, 0]]) * axlen, [255, 0, 0], width=wid)
        self.add_lines(np.array([[0, 0, 0], [0, 1, 0]]) * axlen, [0, 255, 0], width=wid)
        self.add_lines(np.array([[0, 0, 0], [0, 0, 1]]) * axlen, [0, 0, 255], width=wid)

    def show(self):

        self.w.setCameraPosition(distance=self.maxDist * 5)
        self.w.show()
        pg.QtGui.QApplication.exec_()
