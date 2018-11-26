# -*- coding: utf-8 -*-

from __future__ import (absolute_import, division, print_function, unicode_literals)

from time import time
import numpy as np

import pkg_resources

import pyqtgraph as pg
# pg.setConfigOptions(imageAxisOrder='row-major')

from pyqtgraph.Qt import uic, QtGui, QtCore

import bornagain
import bornagain.external.pyqtgraph as bpg
from bornagain.fileio.getters import FrameGetter
from bornagain import analysis

padviewui = pkg_resources.resource_filename('bornagain.viewers.qtviews', 'padview.ui')


class PADView(object):

    r"""
    This is supposed to be an easy way to view PAD data, particularly if you have multiple
    detector panels.  You can set it up by providing a list of :class:`PADGeometry` instances
    along with a list of data arrays.

    It is a work in progress...
    """

    # Note that most of the interface was created using the QT Designer tool.  Here are some important ones:
    # self.histogram is a bornagain.external.pyqtgraph.MultiHistogramLUTWidget

    logscale = False
    pad_geometry = []
    pad_data = []
    pad_labels = None
    mask_data = None
    mask_images = None
    mask_color = None
    _mask_rois = None
    images = None
    scatter_plots = None
    rings = []
    grid = None
    coord_axes = None
    scan_arrows = None
    frame_getter = FrameGetter()
    _px_mode = False
    _shortcuts = None
    _status_string_mouse = ""
    _status_string_getter = " Frame 1 of 1 | "
    evt = None
    show_true_fast_scans = False
    peak_finders = None
    data_filters = None
    show_peaks = True

    def __init__(self, pad_geometry=None, pad_data=None, mask_data=None, logscale=False, frame_getter=None):

        self.logscale = logscale
        self.mask_data = mask_data

        if frame_getter is None:
            if pad_geometry is None or pad_data is None:
                raise ValueError("Either provide a FrameGetter or provide PAD geometry and PAD data."
                                 "One of these items is None type.")
            self.pad_geometry = pad_geometry
            self.pad_data = pad_data
        else:
            self.frame_getter = frame_getter
            self.pad_geometry = frame_getter.pad_geometry
            dat = frame_getter.get_frame(0)
            # while dat is None:
            #     print('searching for data...')
            #     dat = frame_getter.get_next_frame()
            if dat is None:
                raise Exception("Can't find any data!")
            self.pad_data = dat['pad_data']

        self.app = pg.mkQApp()
        self.main_window = uic.loadUi(padviewui)
        self.viewbox = pg.ViewBox()
        self.viewbox.invertX()
        self.viewbox.setAspectLocked()
        self.main_window.graphics_view.setCentralItem(self.viewbox)
        self.setup_pads()
        self.setup_masks()
        self.setup_histogram_tool()
        self.main_window.show()
        # self.label = pg.LabelItem(justify='right')
        # self.viewbox.addItem(self.label)
        self._setup_mouse_interactions()
        self._setup_shortcuts()

        self.main_window.statusbar.setStyleSheet("background-color:rgb(30, 30, 30);color:rgb(255,0,255);"
                                                 "font-weight:bold;font-family:monospace;")

        self.show_frame()

        self.main_window.setWindowState(self.main_window.windowState() & ~pg.QtCore.Qt.WindowMinimized
                                        | pg.QtCore.Qt.WindowActive)
        #self.main_window.activateWindow()
        #self.main_window.showMaximized()

    def _setup_mouse_interactions(self):

        self.proxy = pg.SignalProxy(self.viewbox.scene().sigMouseMoved, rateLimit=60, slot=self._mouse_moved)

    def _setup_menu(self):

        self.main_window.actionGrid.triggered.connect(self.toggle_grid)
        self.main_window.actionRings.triggered.connect(self.edit_ring_radii)

    def _set_simple_keyboard_shortcut(self, key, func):

        if self._shortcuts is None:
            self._shortcuts = []

        self._shortcuts.append(QtGui.QShortcut(QtGui.QKeySequence(key), self.main_window).activated.connect(func))

    def _setup_shortcuts(self):

        self._set_simple_keyboard_shortcut(QtCore.Qt.Key_Right, self.show_next_frame)
        self._set_simple_keyboard_shortcut(QtCore.Qt.Key_Left, self.show_previous_frame)
        self._set_simple_keyboard_shortcut("f", self.show_next_frame)
        self._set_simple_keyboard_shortcut("b", self.show_previous_frame)
        self._set_simple_keyboard_shortcut("r", self.show_random_frame)
        self._set_simple_keyboard_shortcut("n", self.show_history_next)
        self._set_simple_keyboard_shortcut("p", self.show_history_previous)
        self._set_simple_keyboard_shortcut("Ctrl+g", self.toggle_all_geom_info)
        self._set_simple_keyboard_shortcut("Ctrl+r", self.edit_ring_radii)
        self._set_simple_keyboard_shortcut("Ctrl+a", self.toggle_coordinate_axes)
        self._set_simple_keyboard_shortcut("Ctrl+l", self.toggle_pad_labels)
        self._set_simple_keyboard_shortcut("Ctrl+s", self.increase_skip)
        self._set_simple_keyboard_shortcut("Shift+s", self.decrease_skip)
        self._set_simple_keyboard_shortcut("m", self.toggle_masks)
        self._set_simple_keyboard_shortcut("t", self.mask_all_rois)


    def _update_status_string(self, frame_number=None, n_frames=None):

        if frame_number is not None and n_frames is not None:
            n = np.int(np.ceil(np.log10(n_frames)))
            strn = ' Frame %%%dd of %%%dd | ' % (n, n)
            self._status_string_getter = strn % (frame_number, n_frames)

        self.main_window.statusbar.showMessage(self._status_string_getter + self._status_string_mouse)

    @property
    def n_pads(self):

        return len(self.pad_geometry)

    def setup_histogram_tool(self):

        self.main_window.histogram.gradient.loadPreset('flame')
        self.main_window.histogram.setImageItems(self.images)

    def add_roi(self, type='rect', pos=None, size=None):

        if type == 'rect':
            if pos is None:
                pos = (0, 0)
            if size is None:
                size = (100, 100)
            roi = pg.RectROI(pos=pos, size=size, centered=True, sideScalers=True)
            roi.addRotateHandle(pos=(0, 1), center=(0.5, 0.5))
            if self._mask_rois is None:
                self._mask_rois = []
            self._mask_rois.append(roi)
            self.viewbox.addItem(roi)
        else:
            pass

    def hide_rois(self):

        if self._mask_rois is not None:
            for roi in self._mask_rois:
                self.viewbox.removeItem(roi)
            self._mask_rois = None

    def increase_skip(self):

        self.frame_getter.skip = 10**np.floor(np.log10(self.frame_getter.skip)+1)

    def decrease_skip(self):

        self.frame_getter.skip = np.max([10**(np.floor(np.log10(self.frame_getter.skip))-1), 1])

    def show_coordinate_axes(self):

        if self.coord_axes is None:
            x = pg.ArrowItem(pos=(30, 0), brush=pg.mkBrush('r'), pxMode=self._px_mode, angle=180, pen=None)
            y = pg.ArrowItem(pos=(0, 30), brush=pg.mkBrush('g'), pxMode=self._px_mode, angle=-90, pen=None)
            z = pg.ScatterPlotItem([0], [0], pen=None, brush=pg.mkBrush('b'), pxMode=self._px_mode, size=15)
            self.coord_axes = [x, y, z]
            self.viewbox.addItem(z)
            self.viewbox.addItem(x)
            self.viewbox.addItem(y)

    def hide_coordinate_axes(self):

        if self.coord_axes is not None:
            for c in self.coord_axes:
                self.viewbox.removeItem(c)
            self.coord_axes = None

    def toggle_coordinate_axes(self):

        if self.coord_axes is None:
            self.show_coordinate_axes()
        else:
            self.hide_coordinate_axes()

    def show_fast_scan_directions(self):

        if self.scan_arrows is None:

            self.scan_arrows = []

            for p in self.pad_geometry:

                f = p.fs_vec.ravel()/p.pixel_size()
                t = p.t_vec.ravel()/p.pixel_size()
                ang = np.arctan2(f[1], f[0])*180/np.pi + 180
                a = pg.ArrowItem(pos=(t[0], t[1]), angle=ang, brush=pg.mkBrush('r'), pen=None, pxMode=False)

                self.scan_arrows.append(a)
                self.viewbox.addItem(a)

    def hide_fast_scan_directions(self):

        if self.scan_arrows is not None:

            for a in self.scan_arrows:

                self.viewbox.removeItem(a)

            self.scan_arrows = None

    def toggle_fast_scan_directions(self):

        if self.scan_arrows is None:
            self.show_fast_scan_directions()
        else:
            self.hide_fast_scan_directions()

    def show_all_geom_info(self):

        self.show_pad_borders()
        self.show_grid()
        self.show_pad_labels()
        self.show_fast_scan_directions()
        self.show_coordinate_axes()

    def hide_all_geom_info(self):

        self.hide_pad_borders()
        self.hide_grid()
        self.hide_pad_labels()
        self.hide_fast_scan_directions()
        self.hide_coordinate_axes()

    def toggle_all_geom_info(self):

        if self.scan_arrows is None:
            self.show_all_geom_info()
        else:
            self.hide_all_geom_info()

    def show_pad_labels(self):

        if self.pad_labels is None:

            self.pad_labels = []

            for i in range(0, self.n_pads):

                lab = pg.TextItem(text="%d" % i, fill=pg.mkBrush('b'), color='y', anchor=(0.5, 0.5))
                g = self.pad_geometry[i]
                fs = g.fs_vec.ravel()*g.n_fs/2
                ss = g.ss_vec.ravel()*g.n_ss/2
                t = g.t_vec.ravel()
                x = (fs[0] + ss[0] + t[0])/g.pixel_size()
                y = (fs[1] + ss[1] + t[1])/g.pixel_size()
                lab.setPos(x, y)
                self.pad_labels.append(lab)
                self.viewbox.addItem(lab)

    def hide_pad_labels(self):

        if self.pad_labels is not None:
            for lab in self.pad_labels:
                self.viewbox.removeItem(lab)
            self.pad_labels = None

    def toggle_pad_labels(self):

        if self.pad_labels is None:
            self.show_pad_labels()
        else:
            self.hide_pad_labels()

    def scale_factor(self):

        return 1/self.pad_geometry[0].pixel_size()

    def _apply_pad_transform(self, im, p):

        # This is really aweful.  I don't know if this is the best way to do the transforms, but it's the best
        # I could do given the fact that I'm not able to track down all the needed info on how all of these
        # transforms are applied.  I can only say that a physicist did not invent this system.

        # 3D basis vectors of panel (length encodes pixel size):
        f = p.fs_vec.ravel().copy()
        s = p.ss_vec.ravel().copy()

        # 3D translation to *center* of corner pixel (first pixel in memory):
        t = p.t_vec.ravel().copy()

        # Normalize all vectors to pixel size.  This is a hack that needs to be fixed later.  Obviously, we cannot
        # show multiple detectors at different distances using this stupid pixel-based convention.
        # ps = p.pixel_size()
        scl = self.scale_factor()
        f *= scl
        s *= scl
        t *= scl

        # Strip off the z component.  Another dumb move.
        f = f[0:2]
        s = s[0:2]
        t = t[0:2]

        # Offset translation since pixel should be centered.
        t += np.array([0.5, -0.5])

        # These two operations set the "home" position of the panel such that the fast-scan direction
        # is along the viewbox "x" axis.  I don't know if this is the corret thing to do -- needs further
        # attention.  I would imagine that pyqtgraph has a better way to do this, but I haven't found the right
        # feature yet.
        im.scale(1, -1)
        im.rotate(-90)

        # Scale the axes.  This is a phony way to deal with detector tilts.  Better than nothing I guess.
        scf = np.sqrt(np.sum(f * f))
        scs = np.sqrt(np.sum(s * s))

        # Note that the *sign* of this scale factor takes care of the fact that 2D rotations alone
        # cannot deal with a transpose operation.  What follows is a confusing hack... must be a better way.
        sign = np.sign(np.cross(np.array([f[0], f[1], 0]), np.array([s[0], s[1], 0]))[2])

        # Here goes the re-scaling
        im.scale(sign * scs, scf)

        # Rotate the axes
        fnorm = f / np.sqrt(np.sum(f * f))
        ang = np.arctan2(fnorm[1], fnorm[0])
        # No, wait, don't rotate the axes... experimentation says to translate first despite the fact that it
        # doesn't make much sense to do things that way.  Maybe I just gave up too soon, but I found that I needed
        # to translate first...

        # Translate the scaled/rotated image.  Turns out we need to flip the sign of the translation vector
        # coordinate corresponding to the axis we flipped.  I don't know why, but I've completely given up on the
        # idea of understanding the many layers of coordinate systems and transformations...
        im.translate(sign * t[1], t[0])
        im.rotate(-sign * ang * 180 / np.pi)

        # Now, one would *think* that we could define a simple matrix transform, and then translate the resulting
        # image (which has been rotated and possibly skewed to imitate a 3D rotation).  I tried and failed.  But
        # It's worthwile to try again, since it would be brilliant if we could just learn how to define these
        # transforms and apply them to all relevant graphics items.
        # trans = QtGui.QTransform()
        # trans.scale(M[0, 0], M[1, 1])
        # trans.translate(t[0], t[1])
        # im.setTransform(trans)

    def _make_mask_rgba(self, mask):

        d = mask
        mask_rgba = np.zeros((d.shape[0], d.shape[1], 4))
        r = np.zeros_like(d)
        r[d == 0] = self.mask_color[0]
        g = np.zeros_like(d)
        g[d == 0] = self.mask_color[1]
        b = np.zeros_like(d)
        b[d == 0] = self.mask_color[2]
        t = np.zeros_like(d)
        t[d == 0] = 255
        mask_rgba[:, :, 0] = r
        mask_rgba[:, :, 1] = g
        mask_rgba[:, :, 2] = b
        mask_rgba[:, :, 3] = t

        return mask_rgba

    def setup_masks(self, mask_data=None):

        if mask_data is not None:
            self.mask_data = mask_data

        if self.mask_data is None:
            self.mask_data = [np.ones_like(d) for d in self.pad_data]

        if self.mask_color is None:
            self.mask_color = np.array([128, 0, 0])

        for i in range(0, self.n_pads):

            d = self.mask_data[i]

#            if True: # Mask fast-scan pixels
#                d[0, 0: int(np.floor(self.pad_geometry[i].n_fs / 2))] = 1

            mask_rgba = self._make_mask_rgba(d)

            im = pg.ImageItem(mask_rgba)

            self._apply_pad_transform(im, self.pad_geometry[i])

            if self.mask_images is None:
                self.mask_images = []

            self.mask_images.append(im)
            self.viewbox.addItem(im)

            self.main_window.histogram.regionChanged()

    def update_masks(self, mask_data=None):

        if mask_data is not None:
            self.mask_data = mask_data

        if self.mask_images is None:
            self.setup_masks()

        for i in range(0, self.n_pads):

            d = self.mask_data[i]

            mask_rgba = self._make_mask_rgba(d)

            self.mask_images[i].setImage(mask_rgba)

    def hide_masks(self):

        if self.mask_images is not None:
            for im in self.mask_images:
                im.setVisible(False)

    def show_masks(self):

        if self.mask_images is not None:
            for im in self.mask_images:
                im.setVisible(True)

    def toggle_masks(self):
        print('toggle masks')

        if self.mask_images is not None:
            for im in self.mask_images:
                print('set visible')
                im.setVisible(not im.isVisible())
        else:
            print('no mask images')

    def mask_all_rois(self):

        noslice = slice(0, 1, None)

        if self._mask_rois is None:
            return

        for roi in self._mask_rois:

            if not roi.mouseHovering:
                continue

            for (ind, im, dat, geom) in zip(range(self.n_pads), self.images, self.pad_data, self.pad_geometry):

                # Using builtin function of pyqtgraph ROI to identify panels associated with the ROI...
                pslice = roi.getArraySlice(dat, im, axes=(0, 1), returnSlice=True)[0]
                if pslice[0] == noslice and pslice[1] == noslice:
                    continue

                # To find pixels in the rectangular ROI, project pixel coordinates onto the two basis vectors of
                # the ROI.  I couldn't figure out how to do this directly with the ROI class methods.
                sides = [roi.size()[1], roi.size()[0]]
                corner = np.array([roi.pos()[0], roi.pos()[1]])
                angle = roi.angle() * np.pi / 180
                pix_pos = (geom.position_vecs() * self.scale_factor()).reshape(geom.n_ss, geom.n_fs, 3)
                pix_pos = pix_pos[:, :, 0:2] - corner
                v1 = np.array([-np.sin(angle), np.cos(angle)])
                v2 = np.array([np.cos(angle), np.sin(angle)])
                ind1 = np.dot(pix_pos, v1)
                ind2 = np.dot(pix_pos, v2)
                ind1 = (ind1 >= 0) * (ind1 <= sides[0])
                ind2 = (ind2 >= 0) * (ind2 <= sides[1])
                inds = ind1*ind2

                # Here comes the mask update for this pad:
                m = self.mask_data[ind]
                m[inds] = 0
                self.mask_data[ind] = m
                self.mask_images[ind].setImage(self._make_mask_rgba(m))

    def setup_pads(self, pad_data=None):

        if pad_data is not None:
            self.pad_data = pad_data

        mx = np.ravel(self.pad_data).max()

        for i in range(0, self.n_pads):

            d = self.pad_data[i]

            if self.logscale:
                d[d < 0] = 0
                d = np.log10(d)

            if self.show_true_fast_scans:  # For testing - show fast scan axis
                d[0, 0:int(np.floor(self.pad_geometry[i].n_fs/2))] = mx

            im = pg.ImageItem(d)

            self._apply_pad_transform(im, self.pad_geometry[i])

            if self.images is None:
                self.images = []

            self.images.append(im)
            self.viewbox.addItem(im)

            self.main_window.histogram.regionChanged()

    def update_pads(self, pad_data=None):

        if pad_data is not None:
            self.pad_data = pad_data

        mx = np.ravel(self.pad_data).max()

        if self.images is None:
            self.setup_pads()

        for i in range(0, self.n_pads):

            d = self.pad_data[i]

            if self.show_true_fast_scans:  # For testing - show fast scan axis
                d[0, 0:int(np.floor(self.pad_geometry[i].n_fs/2))] = mx

            if self.logscale:
                d[d < 0] = 0
                d = np.log10(d)

            self.images[i].setImage(d)

        self.main_window.histogram.regionChanged()

    def _mouse_moved(self, evt):

        if evt is None:
            return

        self.evt = evt
        pos = evt[0]
        pid = -1
        ppos = (-1, -1)
        intensity = -1
        if self.viewbox.sceneBoundingRect().contains(pos):
            for i in range(0, len(self.images)):
                if self.images[i].sceneBoundingRect().contains(pos):
                    pid = i
                    ppos = self.images[i].mapFromScene(pos)
                    ppos = (np.floor(ppos.x()), np.floor(ppos.y()))
                    continue

            # pnt = self.viewbox.mapSceneToView(pos)
            fs = np.int(ppos[1])
            ss = np.int(ppos[0])
            if pid >= 0:
                d = self.pad_data[pid]
                sh = d.shape
                if ss < sh[0] and fs < sh[1]:
                    intensity = self.pad_data[pid][ss, fs]

            if pid >= 0:
                self._status_string_mouse = ' Panel %2d  |  Pixel %4d,%4d  |  Value=%8g  | ' % (pid, ss, fs, intensity)
            else:
                self._status_string_mouse = ''

            self._update_status_string()
            #self.label.setText(
            #    "<span style='font-size: 12pt'>x=%0.1f,   <span style='color: red'>y1=%0.1f</span>" % (
            #    pnt.x(), pnt.y()))

    def edit_ring_radii(self):

        text, ok = QtGui.QInputDialog.getText(self.main_window, "Enter ring radii (comma separated)", "Ring radii",
                                              QtGui.QLineEdit.Normal, "100,200")
        if ok:
            if text == '':
                self.remove_rings()
                return
            r = text.split(',')
            rad = []
            for i in range(0, len(r)):
                try:
                    rad.append(float(r[i].strip()))
                except:
                    pass
            self.remove_rings()
            self.add_rings(rad)

    def add_rings(self, radii=[], pens=None, radius_handle=False):

        if not isinstance(radii, (list,)):
            radii = [radii]

        n = len(radii)

        if pens is None:
            pens = [pg.mkPen([255, 255, 255], width=2)]*n

        for i in range(0, n):
            circ = pg.CircleROI(pos=[-radii[i]*0.5]*2, size=radii[i], pen=pens[i])
            circ.translatable = False
            circ.removable = True
            self.rings.append(circ)
            self.viewbox.addItem(circ)

        if not radius_handle:

            self.hide_ring_radius_handles()

    def hide_ring_radius_handles(self):

        for circ in self.rings:
            for handle in circ.handles:
                circ.removeHandle(handle['item'])

    def remove_rings(self):

        if self.rings is None:
            return

        for i in range(0, len(self.rings)):
            self.viewbox.removeItem(self.rings[i])

    def show_grid(self):

        if self.grid is None:
            self.grid = pg.GridItem()
        self.viewbox.addItem(self.grid)

    def hide_grid(self):

        if self.grid is not None:
            self.viewbox.removeItem(self.grid)
            self.grid = None

    def toggle_grid(self):

        if self.grid is None:
            self.show_grid()
        else:
            self.hide_grid()

    def show_pad_border(self, n, pen=None):

        if pen is None:
            pen = pg.mkPen([0, 255, 0], width=1)
        self.images[n].setBorder(pen)

    def show_pad_borders(self, pen=None):

        if pen is None:
            pen = pg.mkPen([0, 255, 0], width=1)
        for image in self.images:
            image.setBorder(pen)

    def hide_pad_borders(self):

        for image in self.images:
            image.setBorder(None)

    def add_scatter_plot(self, *args, **kargs):

        if self.scatter_plots is None:
            self.scatter_plots = []

        scat = pg.ScatterPlotItem(*args, **kargs)
        self.scatter_plots.append(scat)
        self.viewbox.addItem(scat)

    def remove_scatter_plots(self):

        if self.scatter_plots is None:
            return

        for scat in self.scatter_plots:
            self.viewbox.removeItem(scat)

        self.scatter_plots = None

    def show_history_next(self):

        if self.frame_getter is None:
            print('no getter')
            return

        dat = self.frame_getter.get_history_next()

        self.load_frame_dat(dat)

    def show_history_previous(self):

        if self.frame_getter is None:
            print('no getter')
            return

        dat = self.frame_getter.get_history_previous()

        self.load_frame_dat(dat)

    def show_next_frame(self):

        if self.frame_getter is None:
            print('no getter')
            return

        dat = self.frame_getter.get_next_frame()

        self.load_frame_dat(dat)

    def show_previous_frame(self):

        if self.frame_getter is None:
            print('no getter')
            return

        dat = self.frame_getter.get_previous_frame()

        self.load_frame_dat(dat)

    def show_random_frame(self):

        dat = self.frame_getter.get_random_frame()

        self.load_frame_dat(dat)

    def show_frame(self, frame_number=0):

        if self.frame_getter is None:
            print('no getter')
            return

        dat = self.frame_getter.get_frame(frame_number=frame_number)

        self.load_frame_dat(dat)

    def load_frame_dat(self, dat):

        r"""

        Update display with new data.

        Args:
            dat: input dictionary with keys 'pad_data', 'peaks'

        Returns:

        """

        self.remove_scatter_plots()

        if dat is None:
            return
            # TODO: handle None more wisely

        if 'pad_data' in dat.keys():

            self.pad_data = dat['pad_data']

            # if bornagain.get_global('debug'):

            # pad_data = dat['pad_data']
            # t = time()
            # pad_data = analysis.peaks.snr_filter_pool(pad_data)
            # print((time() - t), ' ms (mp)')
            #
            # pad_data = dat['pad_data']
            # t = time()
            # for i in range(0, len(pad_data)):
            #     pad_data[i] = analysis.peaks.snr_filter(pad_data[i])
            # print((time() - t), ' ms')

            if self.data_filters is not None:
                t = time()

                for filter in self.data_filters:
                    filter(self)
                # pad_data = [self.data_filters(d) for d in pad_data]
                # pad_data2 = []
                # for i in range(0, len(pad_data)):
                #     # print('panel', i, pad_data[i].shape)
                #     pad_data2.append(self.data_filters(pad_data[i]))
                print(time()-t)
                # pad_data = pad_data2
            # else:
            #     pad_data = dat['pad_data']
            #     print(type(pad_data))
            #     if hasattr(self.data_filters, '__call__'):
            #         t = time()
            #         for i in range(0, len(pad_data)):
            #             pad_data[i] = self.data_filters(pad_data[i])
            #         print((time() - t)/len(pad_data), ' ms')

            #self.update_pads(pad_data)
            self.update_pads()

        # if self.peak_finders is not None:
        #
        #     pads = dat['pad_data']
        #
        #     new_pads = []
        #
        #     for pad, peak_finder in zip(pads, self.peak_finders):
        #         # new_pads.append(peak_finder.get_signal_above_background(pad))
        #         new_pads.append(peak_finder.get_snr(pad))
        #
        #     self.update_pads(new_pads)
        #

        if self.show_peaks is True:

            if 'peaks' in dat.keys():

                peaks = dat['peaks']
                if peaks is not None:
                    n_peaks = peaks['n_peaks']
                    pad_numbers = peaks['pad_numbers']
                    fs_pos = peaks['fs_pos']
                    ss_pos = peaks['ss_pos']

                    pad_geom = self.pad_geometry
                    gl_fs_pos = np.zeros(n_peaks)
                    gl_ss_pos = np.zeros(n_peaks)
                    for i in range(0, n_peaks):
                        pad_num = pad_numbers[i]
                        vec = pad_geom[pad_num].indices_to_vectors(ss_pos[i], fs_pos[i]).ravel()
                        gl_fs_pos[i] = vec[0]*self.scale_factor()
                        gl_ss_pos[i] = vec[1]*self.scale_factor()
                    self.add_scatter_plot(gl_fs_pos, gl_ss_pos, pen=pg.mkPen('g'), brush=None, width=5, size=10, pxMode=False)

        self._update_status_string(frame_number=self.frame_getter.current_frame, n_frames=self.frame_getter.n_frames)

        self._mouse_moved(self.evt)

    def start(self):

        self.show_frame(0)
        self.app.exec_()
