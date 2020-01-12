from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np
from bornagain.utils import warn_pyqtgraph
try:
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl
    from pyqtgraph.Qt import QtGui
    # Some default bright colors.  Might need to make this list longer in the future.
    colors = [pg.glColor([255, 0, 0]),
              pg.glColor([0, 255, 0]),
              pg.glColor([0, 0, 255]),
              pg.glColor([255, 255, 0]),
              pg.glColor([0, 255, 255]),
              pg.glColor([255, 0, 255]),
              pg.glColor([255, 255, 255]),
              pg.glColor([255, 128, 128])]

    pens = [pg.mkPen([255, 0, 0]),
            pg.mkPen([0, 255, 0]),
            pg.mkPen([0, 0, 255]),
            pg.mkPen([255, 255, 0]),
            pg.mkPen([0, 255, 255]),
            pg.mkPen([255, 0, 255]),
            pg.mkPen([255, 255, 255]),
            pg.mkPen([255, 128, 128])]
except ImportError:
    warn_pyqtgraph()

"""
This is supposed to have various viewers that use pyqtgraph.  It's mostly useless right now.
"""


def bright_colors(i, alpha=1):

    """ Some nice colors.  Only 8 available, which loops around as the input index increments."""

    col = list(colors[i % len(colors)])
    col[3] = alpha
    return col


class Volumetric3D(object):
    ''' View a 3D density map '''

    def __init__(self):

        self.app = pg.mkQApp() #QtGui.QApplication([])
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

    def show(self, smooth=True, hold=True, kill=True):

        self.dat[..., 0:-1] *= 255. / self.dat[..., 0:-1].max()
        self.dat[..., 3] *= 255. / self.dat[..., 3].max()
        self.dat = np.ubyte(self.dat)
        v = gl.GLVolumeItem(self.dat, smooth=smooth)
        v.translate(-self.dat.shape[0] / 2., -self.dat.shape[1] / 2., -self.dat.shape[2] / 2.)

        self.w.addItem(v)
        self.w.setCameraPosition(distance=self.maxDist * 2)
        self.w.show()
        if hold:
            self.app.exec_()
        if kill:
            del self.app


def plot_multiple_images(images, title=None, n_rows=None, hold=True, kill=True):

    app = pg.mkQApp()
    win = pg.GraphicsLayoutWidget()
    if title is not None:
        win.setWindowTitle(title)
    plots = []
    image_items = []
    for i in range(len(images)):
        plot = win.addPlot()
        plot.setAspectLocked()
        plots.append(plot)
        img = pg.ImageItem(images[i])
        image_items.append(img)
        plot.addItem(img)
        plot.setXRange(0, np.max(images[i].shape))
        plot.setYRange(0, np.max(images[i].shape))
        if n_rows is not None:
            if (i % np.ceil(len(images)/n_rows)) == 0:
                win.nextRow()
    win.show()
    if hold:
        app.exec_()
    if kill:
        del app


def MapProjection(data, axis=None, hold=True, kill=True, title=None):
    r""" View a 3D density map as a projection along selected axes (which can be a list) """
    if axis is None:
        axis = [0, 1, 2]
    if type(axis) is not list:
        axis = [axis]
    dat = []
    for ax in axis:
        dat.append(np.sum(data, axis=ax))
    plot_multiple_images(dat, hold=hold, kill=kill, title=title)


def MapSlices(data, axis=None, levels=None, hold=True, kill=True, title=None):
    r""" View a 3D density map as a projection along selected axes (which can be a list) """

    if axis is None:
        axis = [0, 1, 2]

    if not type(axis) is list:
        axis = [axis]

    dat = []

    for ax in axis:
        if ax == 0:
            dat.append(np.squeeze(data[int(np.floor(data.shape[0]/2)), :, :]))
        elif ax == 1:
            dat.append(np.squeeze(data[:, int(np.floor(data.shape[1]/2)), :]))
        elif ax == 2:
            dat.append(np.squeeze(data[:, :, int(np.floor(data.shape[2]/2))]))

    plot_multiple_images(dat, hold=hold, kill=kill, title=title)


class Scatter3D(object):

    r''' Simple viewer for 3D scatter plots. '''

    app = None

    def __init__(self, title=None):

        self.app = pg.mkQApp() #QtGui.QApplication([])
        self.w = gl.GLViewWidget()
        self.w.window().setWindowTitle(title)
        self.defaultColor = pg.glColor([255, 255, 255])
        self.defaultSize = 1
        self.defaultWidth = 1
        self.maxDist = 0
        self.orthographic = False

    def __del__(self):
        r'''
        Delete reference C++ will call destructor.
        '''
        if self.app is not None:
            del self.app

    def add_points(self, r, color=None, size=None):

        r'''
        Add an Nx3 array of points r with specified color and size.  Color is a 3-element
        Python list and size is a float scalar.
        '''

        if color is None:
            col = self.defaultColor
        else:
            col = color #pg.glColor(color)

        if size is None:
            siz = self.defaultSize
        else:
            siz = size

        self.maxDist = max(self.maxDist, np.amax(np.sqrt(np.sum(r * r, axis=-1))))
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

    def add_unit_cell(self, cell, width=None):

        if width is None:
            wid = self.defaultWidth
        else:
            wid = width

        a = cell.a_vec
        b = cell.b_vec
        c = cell.c_vec

        self.add_lines(np.array([[0, 0, 0], a]), [255, 0, 0], width=wid)
        self.add_lines(np.array([[0, 0, 0], b]), [0, 255, 0], width=wid)
        self.add_lines(np.array([[0, 0, 0], c]), [0, 0, 255], width=wid)
        self.add_lines(np.array([[0, 0, 0], a]) + b, [255, 255, 255], width=wid)
        self.add_lines(np.array([[0, 0, 0], a]) + c, [255, 255, 255], width=wid)
        self.add_lines(np.array([[0, 0, 0], a]) + b + c, [255, 255, 255], width=wid)
        self.add_lines(np.array([[0, 0, 0], b]) + a, [255, 255, 255], width=wid)
        self.add_lines(np.array([[0, 0, 0], b]) + c, [255, 255, 255], width=wid)
        self.add_lines(np.array([[0, 0, 0], b]) + a + c, [255, 255, 255], width=wid)
        self.add_lines(np.array([[0, 0, 0], c]) + a, [255, 255, 255], width=wid)
        self.add_lines(np.array([[0, 0, 0], c]) + b, [255, 255, 255], width=wid)
        self.add_lines(np.array([[0, 0, 0], c]) + a + b, [255, 255, 255], width=wid)

    def set_orthographic_projection(self):

        self.orthographic = True
        dist = self.maxDist*100000
        fov = 200*self.maxDist/dist
        print(fov, dist)
        self.w.opts['distance'] = dist
        self.w.opts['fov'] = fov

    def show(self, hold=True, kill=True):

        if not self.orthographic:
            self.w.setCameraPosition(distance=self.maxDist * 5)
        self.w.show()
        if hold:
            self.app.exec_()
        if kill:
            del self.app


if __name__ == '__main__':

    images = [np.random.rand(5, 5), np.random.rand(5, 6), np.random.rand(5, 10)]
    images[0][0:2, 0:2] = -1
    plot_multiple_images(images, title='test', n_rows=2)
