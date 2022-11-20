# This file is part of reborn <https://kirianlab.gitlab.io/reborn/>.
#
# reborn is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# reborn is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with reborn.  If not, see <https://www.gnu.org/licenses/>.


import weakref
import pyqtgraph as pg


def keep_open():
    r"""
    Simple helper that keeps a pyqtgraph window open when you run a script from the terminal.
    """
    pg.QtGui.QApplication.instance().exec_()


images = []


def image(*args, **kargs):
    """
    Create and return an :class:`ImageWindow <pyqtgraph.ImageWindow>`
    (this is just a window with :class:`ImageView <pyqtgraph.ImageView>` widget inside), show image data inside.
    Will show 2D or 3D image data.
    Accepts a *title* argument to set the title of the window.
    All other arguments are used to show data. (see :func:`ImageView.setImage() <pyqtgraph.ImageView.setImage>`)

    Note:
        This is a direct copy of the equivalent (undocumented?) function in |pyqtgraph|, except that it adds the
        *view* argument so that you can easily put your image in a plot axis.
    """
    pg.mkQApp()
    w = ImageWindow(*args, **kargs)
    images.append(w)
    w.show()
    return w


def imview(image, *args, hold=False, **kwargs):
    r""" Makes an ImageView widget but adds some functionality that I commonly need.  A PlotItem is used as the
    view so that there are axes, and by default the aspect ratio is unlocked so that the image is stretched to fill
    the view.  The default colormap is set to 'flame'. The hold option keeps the window open, so that your script
    will not exit and close your window a few milliseconds after opening it.  A Qt application is created if need be,
    which avoids a common runtime error. """
    app = pg.mkQApp()
    imv = ImView(*args, image, gradient='flame', aspect_locked=False, **kwargs)
    # plot.setAspectLocked(False)
    # imv.setImage(image)
    # imv.setPredefinedGradient('flame')
    imv.show()
    if hold:
        app.exec_()
    return imv


class ImView(pg.ImageView):

    plots = []
    lines = []
    rois = []
    first = True

    def __init__(self, *args, **kwargs):
        r"""
        A subclass of pyqtgraph.ImageView that adds helpful features when you need the image to be embedded in a
        plot axis.  Lists of roi, plot, and line items are maintained.
        In addition to the usual positional and keyword arguments for the pyqtgraph.ImageView class, the following
        arguments are accepted:

        Arguments:
            gradient (str): Set the colormap (‘thermal’, ‘flame’, ‘yellowy’, ‘bipolar’, ‘spectrum’, ‘cyclic’,
                            ‘greyclip’, ‘grey’, ‘viridis’, ‘inferno’, ‘plasma’, ‘magma’)
            show_histogram (bool): Set to False if you don't want the usual histogram to appear on the right.
            aspect_locked (bool): Set to False if you don't want the aspect ratio of the plot/image to be fixed to 1.
            title (str): Plot title.
            ss_label (str): X (horizontal) axis label.
            fs_label (str): Y (vertical) axis label.
            fs_lims (tuple of float): The min and max values of pixel coordinates along fast-scan axis.
            ss_lims (tuple of float): The min and max values of pixel coordinates along slow-scan axis.
        """
        gradient = kwargs.pop('gradient', 'grey') #'flame')
        show_histogram = kwargs.pop('show_histogram', True)
        aspect_locked = kwargs.pop('aspect_locked', True)
        title = kwargs.pop('title', None)
        ss_label = kwargs.pop('ss_label', 'Axis 1')
        fs_label = kwargs.pop('fs_label', 'Axis 0')
        self.fs_lims = kwargs.pop('fs_lims', None)
        self.ss_lims = kwargs.pop('ss_lims', None)
        self.frame_names = kwargs.pop('frame_names', None)
        self.app = pg.mkQApp()
        p = pg.PlotItem()
        super().__init__(view=p)
        self.view.setAspectLocked(aspect_locked)
        self.view.invertY(False)
        if title is not None:
            self.view.setTitle(title)
        if fs_label is not None:
            self.view.setLabel('left', fs_label)
        if ss_label is not None:
            self.view.setLabel('bottom', ss_label)
        self.ui.roiBtn.hide()
        self.ui.menuBtn.hide()
        if len(args) > 0:
            self.setImage(*args, **kwargs)
        self.setPredefinedGradient(gradient)
        if show_histogram:
            self.ui.histogram.show()
        else:
            # self.ui.histogram = pg.GradientEditorItem()
            self.ui.histogram.hide()
        self._frame_changed(0, None)
        self.sigTimeChanged.connect(self._frame_changed)

    def setImage(self, *args, **kwargs):
        r""" Override setImage so that the image is positioned properly on the plot.  The default puts the *corner*
        of the first pixel at the position (0, 0) by default, which makes no sense; the *center* of the pixel should
        be located at (0, 0) for all the obvious reasons.

        In addition to the usual positional and keyword arguments, the following keyword arguments are accepted:

        Arguments:
            fs_lims (list): The minimum and maximum values of the pixel coordinates along the fast-scan direction.
            ss_lims (list): The minimum and maximum values of the pixel coordinates along the slow-scan direction.

        """
        image = args[0]
        if self.first:
            ns, nf = image.shape[-2:]
            self.fs_lims = kwargs.pop('fs_lims', self.fs_lims)
            self.ss_lims = kwargs.pop('ss_lims', self.ss_lims)
            if self.fs_lims is None:
                self.fs_lims = np.array([0, nf-1])
            if self.ss_lims is None:
                self.ss_lims = np.array([0, ns-1])
            fs_width = self.fs_lims[1] - self.fs_lims[0]
            ss_width = self.ss_lims[1] - self.ss_lims[0]
            fs_scale = fs_width/(nf-1)
            ss_scale = ss_width/(ns-1)
            kwargs.setdefault('scale', (ss_scale, fs_scale))
            kwargs.setdefault('pos', (self.ss_lims[0]-0.5*ss_scale, self.fs_lims[0]-0.5*fs_scale))
            self.first = False
        super().setImage(*args, autoRange=False, **kwargs)

    def add_plot(self, *args, **kwargs):
        r""" TODO: Document. """
        name = kwargs.pop('name', None)
        plot = self.getView().plot(*args, **kwargs)
        plot.name = name
        self.plots.append(plot)
        return plot

    def add_line(self, *args, position=None, vertical=False, horizontal=False, **kwargs):
        r""" Adds pyqtgraph.InfiniteLine.  Accepts the InfiniteLine arguments:
        pos=None, angle=90, pen=None, movable=False, bounds=None, hoverPen=None, label=None, labelOpts=None,
        span=(0, 1), markers=None, name=None

        Adds the following arguments:
            horizontal (bool): Make the line horizontal (angle=)
            vertical (bool): Make the line vertical (angle=)
            position (str): Same as pos argument for InfiniteLine

        """
        if vertical:
            kwargs['angle'] = 90
        if horizontal:
            kwargs['angle'] = 0
        line = pg.InfiniteLine(*args, **kwargs)
        if position is not None:
            line.setPos([position, position])
        self.lines.append(line)
        self.getView().addItem(line)
        return line

    def add_roi(self, *args, **kwargs):
        r""" TODO: Document. """
        nx, ny = self.image.shape[-2:]
        name = kwargs.pop('name', None)
        pos = kwargs.pop('pos', (nx/4, ny/4))
        size = kwargs.pop('size', (nx/2, ny/2))
        pen = kwargs.pop('pen', pg.mkPen('r', width=5))
        roi = pg.ROI(*args, pos=pos, size=size, pen=pen, **kwargs)
        roi.addScaleHandle([1, 1], [0, 0])
        roi.addScaleHandle([1, 0], [1, 0])
        roi.addScaleHandle([1, 0], [0, 1])
        roi.addScaleHandle([0, 1], [1, 0])
        roi.name = name
        self.rois.append(roi)
        self.getView().addItem(roi)
        return roi

    def get_roi(self, name=None):
        roi = None
        for r in self.rois:
            if r.name == name:
                roi = r
                break
        if roi is None:
            roi = self.add_roi(name=name)
        return roi

    def get_roi_slice(self, name=None):
        roi = self.get_roi(name)
        (nx, ny) = self.image.shape[-2:]
        (px, py) = roi.pos()
        (sx, sy) = roi.size()
        slc = np.s_[max(int(px), 0):min(int(px+sx), nx - 1), max(int(py), 0):min(int(py+sy), ny - 1)]
        return slc

    def get_plot(self, name=None):
        plot = None
        for p in self.plots:
            if p.name == name:
                plot = p
                break
        if plot is None:
            plot = self.add_plot(name=name)
        return plot

    def get_line(self, name=None):
        line = None
        for l in self.lines:
            if l.name == name:
                line = l
                break
        if line is None:
            line = self.add_line(name=name)
        return line

    def set_mask(self, mask, color=None):
        d = mask
        if color is None:
            color = (255, 255, 255, 20)
        mask_rgba = np.zeros((d.shape[0], d.shape[1], 4))
        r = np.zeros_like(d)
        r[d == 0] = color[0]
        g = np.zeros_like(d)
        g[d == 0] = color[1]
        b = np.zeros_like(d)
        b[d == 0] = color[2]
        t = np.zeros_like(d)
        t[d == 0] = color[3]
        mask_rgba[:, :, 0] = r
        mask_rgba[:, :, 1] = g
        mask_rgba[:, :, 2] = b
        mask_rgba[:, :, 3] = t
        im = pg.ImageItem(mask_rgba)
        self.getView().addItem(im)

    # def keyPressEvent(self, ev):
    #     if ev.key() == 82:  # r button
    #         print("'r' pushed")
    #         self.retry = True
    #         global retry
    #         retry = True
    #         self.app.quit()
    #     if ev.key() == 32:  # spacebar
    #         print("spacebar pushed")
    #         self.app.quit()
    #     if ev.key() == 81:  # q button
    #         print("'q' pushed")
    #         global quit
    #         quit = True
    #         self.quit = True
    #         self.app.quit()
    #     if ev.key() == 75:
    #         print("'k' pushed")
    #         global kill
    #         kill = True
    #         self.app.closeAllWindows()
    #         self.app.quit()
    #         self.kill = True
    #         del (self.app)

    def _frame_changed(self, ind, time):
        if self.frame_names is not None:
            self.view.setTitle(self.frame_names[ind])
    #     # TODO make two versions of this, onewith jet and one with ntip
    #     # check setVisible(True or False)
    #     if (self.ntip_idx is None) or (self.jet_breakup_idx is None):
    #         return
    #     p = self.ntip_idx[ind]
    #     d = self.jet_breakup_idx[ind]
    #     dc = self.drop_centroids[ind]
    #     if self.ntip_position is None:
    #         self.ntip_position = self.add_line(angle=90, position=p, movable=True, pen='r')
    #         self.jtip_position = self.add_line(angle=90, position=d, movable=True, pen='b')
    #         self.drop_plot = self.add_plot(pen=None, symbolPen='g', symbol='o', symbolBrush=None, symbolSize=10)
    #         self.drop_plot.setData(dc[:,0],dc[:,1])
    #     else:
    #         self.ntip_position.setPos([p, p])
    #         self.jtip_position.setPos([d,d])
    #         self.drop_plot.setData(dc[:,0],dc[:,1])




class ImageWindow(pg.ImageView):
    """
    (deprecated; use ImageView instead)

    Note:
        This is a direct copy of the equivalent (undocumented?) class in |pyqtgraph|, except that it adds the
        *view* argument so that you can easily put your image in a plot axis.
    """
    def __init__(self, *args, **kargs):
        pg.mkQApp()
        self.win = pg.QtGui.QMainWindow()
        self.win.resize(800, 600)
        if 'title' in kargs:
            self.win.setWindowTitle(kargs['title'])
            del kargs['title']
        view = None
        fs_lims = None
        ss_lims = None
        if 'view' in kargs:
            view = kargs['view']
            if view == 'plot':
                view = pg.PlotItem()
            del kargs['view']
            if 'fs_lims' in kargs:
                fs_lims = kargs['fs_lims']
            if 'ss_lims' in kargs:
                ss_lims = kargs['ss_lims']
        if 'fs_lims' in kargs:
            del kargs['fs_lims']
        if 'ss_lims' in kargs:
            del kargs['ss_lims']
        if view is not None:
            img = args[0]
            if fs_lims is not None and ss_lims is not None:
                fs_width = fs_lims[1] - fs_lims[0]
                ss_width = ss_lims[1] - ss_lims[0]
                dfs = fs_width/img.shape[1]
                dss = ss_width/img.shape[0]
                kargs['pos'] = (ss_lims[0]-0.5*dss, fs_lims[0]-0.5*dfs)
                kargs['scale'] = (ss_width/(img.shape[0]-1), fs_width/(img.shape[1]-1))
        pg.ImageView.__init__(self, self.win, view=view)
        self.view.setAspectLocked(False)
        self.view.enableAutoRange(True)
        self.view.invertY(False)
        if len(args) > 0 or len(kargs) > 0:
            self.setImage(*args, **kargs)
        self.win.setCentralWidget(self)
        for m in ['resize']:
            setattr(self, m, getattr(self.win, m))
        self.win.show()


class MultiHistogramLUTWidget(pg.GraphicsView):

    r"""
    This is the equivalent of :class:`pyqtgraph.HistogramLUTWidget`, but wraps
    :class:`MultiHistogramLUTWidget <reborn.external.pyqtgraph.MultiHistogramLUTItem>` instead of
    :class:`pyqtgraph.HistogramLUTItem`.
    """

    def __init__(self, parent=None, *args, **kargs):
        background = kargs.get('background', 'default')
        pg.GraphicsView.__init__(self, parent, useOpenGL=False, background=background)
        self.item = MultiHistogramLUTItem(*args, **kargs)
        self.setCentralItem(self.item)
        self.setSizePolicy(pg.QtGui.QSizePolicy.Preferred, pg.QtGui.QSizePolicy.Expanding)
        self.setMinimumWidth(100)

    def sizeHint(self):
        r"""
        Undocumented pyqtgraph method.
        """
        return pg.QtCore.QSize(115, 200)

    def __getattr__(self, attr):
        return getattr(self.item, attr)


class MultiHistogramLUTItem(pg.HistogramLUTItem):

    r"""
    This is a reborn extension to the :class:`pyqtgraph.HistogramLUTItem` that allows control
    over multiple images. The main feature is the
    addition of the :func:`setImageItems` method.

    This is a :class:`pyqtgraph.graphicsWidget` which provides controls for adjusting the display of an image.

    Includes:

    - Image histogram
    - Movable region over histogram to select black/white levels
    - Gradient editor to define color lookup table for single-channel images

    Parameters
    ----------
    image : ImageItem or None
        If *image* is provided, then the control will be automatically linked to
        the image and changes to the control will be immediately reflected in
        the image's appearance.
    fillHistogram : bool
        By default, the histogram is rendered with a fill.
        For performance, set *fillHistogram* = False.
    rgbHistogram : bool
        Sets whether the histogram is computed once over all channels of the
        image, or once per channel.
    levelMode : 'mono' or 'rgba'
        If 'mono', then only a single set of black/whilte level lines is drawn,
        and the levels apply to all channels in the image. If 'rgba', then one
        set of levels is drawn for each channel.
    """

    def __init__(self, *args, **kwargs):
        r"""
        Undocumented pyqtgraph method.
        """

        pg.HistogramLUTItem.__init__(self, *args, **kwargs)

    def setImageItems(self, imgs):
        r"""
        Set a list of :class:`pyqtgraph.ImageItem` instances that will have their levels
        and LUT automatically controlled by this HistogramLUTItem.
        """
        i = 0
        self.imageItems = []
        for img in imgs:
            self.imageItems.append(img)
            img.sigImageChanged.connect(self.imageChanged)
            # send function pointer, not the result
            img.setLookupTable(self.getLookupTable)
            i += 1

        phonyarray = np.atleast_2d(np.concatenate([im.image.ravel() for im in imgs]))  # np.ravel([im.image for im in imgs])

        #phonyarray = phonyarray[0:(len(phonyarray) - (len(phonyarray) % 2))]
        #phonyarray = phonyarray.reshape([2, int(len(phonyarray) / 2)])

        self.imageItemStrong = pg.ImageItem(phonyarray)
        self.imageItem = weakref.ref(self.imageItemStrong)  # TODO: fix this up
        self.regionChanged()
        self.imageChanged(autoLevel=False)

    def gradientChanged(self):
        r"""
        Undocumented pyqtgraph method.
        """

        if self.imageItem() is not None:
            if self.gradient.isLookupTrivial():
                for im in self.imageItems:
                    im.setLookupTable(None)  # lambda x: x.astype(np.uint8))
            else:
                for im in self.imageItems:
                    # send function pointer, not the result
                    im.setLookupTable(self.getLookupTable)

        self.lut = None
        self.sigLookupTableChanged.emit(self)

    def regionChanged(self):
        r"""
        Undocumented pyqtgraph method.
        """

        if self.imageItem() is not None:
            # print('regionChanged')
            for im in self.imageItems:
                im.setLevels(self.getLevels())
        self.sigLevelChangeFinished.emit(self)

    def regionChanging(self):
        r"""
        Undocumented pyqtgraph method.
        """
        if self.imageItem() is not None:
            # print('regionChanging')
            for im in self.imageItems:
                im.setLevels(self.getLevels())
        self.sigLevelsChanged.emit(self)
        self.update()

    def setLevelMode(self, mode):
        r"""
        Set the method of controlling the image levels offered to the user.
        Options are 'mono' or 'rgba'.
        """
        assert mode in ('mono', 'rgba')

        print('setLevelMode')

        if mode == self.levelMode:
            return

        oldLevels = self.getLevels()
        self.levelMode = mode
        self._showRegions()

        # do our best to preserve old levels
        if mode == 'mono':
            levels = np.array(oldLevels).mean(axis=0)
            self.setLevels(*levels)
        else:
            levels = [oldLevels] * 4
            self.setLevels(rgba=levels)

        # force this because calling self.setLevels might not set the imageItem
        # levels if there was no change to the region item
        self.imageItem().setLevels(self.getLevels())
        for im in self.imageItems:
            im.setLevels(self.getLevels())

        self.imageChanged()
        self.update()


from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import collections
from pyqtgraph import functions as fn
from pyqtgraph import debug as debug
from pyqtgraph import GraphicsObject
from pyqtgraph import Point
from pyqtgraph import getConfigOption


class ImageItem(GraphicsObject):
    """
    **Bases:** :class:`GraphicsObject <pyqtgraph.GraphicsObject>`

    GraphicsObject displaying an image. Optimized for rapid update (ie video display).
    This item displays either a 2D numpy array (height, width) or
    a 3D array (height, width, RGBa). This array is optionally scaled (see
    :func:`setLevels <pyqtgraph.ImageItem.setLevels>`) and/or colored
    with a lookup table (see :func:`setLookupTable <pyqtgraph.ImageItem.setLookupTable>`)
    before being displayed.

    ImageItem is frequently used in conjunction with
    :class:`HistogramLUTItem <pyqtgraph.HistogramLUTItem>` or
    :class:`HistogramLUTWidget <pyqtgraph.HistogramLUTWidget>` to provide a GUI
    for controlling the levels and lookup table used to display the image.
    """

    xds = 1
    yds = 1

    sigImageChanged = QtCore.Signal()
    sigRemoveRequested = QtCore.Signal(object)  # self; emitted when 'remove' is selected from context menu

    def __init__(self, image=None, **kargs):
        """
        See :func:`setImage <pyqtgraph.ImageItem.setImage>` for all allowed initialization arguments.
        """
        GraphicsObject.__init__(self)
        self.menu = None
        self.image = None  ## original image data
        self.qimage = None  ## rendered image for display

        self.paintMode = None

        self.levels = None  ## [min, max] or [[redMin, redMax], ...]
        self.lut = None
        self.autoDownsample = False

        self.axisOrder = getConfigOption('imageAxisOrder')

        # In some cases, we use a modified lookup table to handle both rescaling
        # and LUT more efficiently
        self._effectiveLut = None

        self.drawKernel = None
        self.border = None
        self.removable = False

        if image is not None:
            self.setImage(image, **kargs)
        else:
            self.setOpts(**kargs)

    def setCompositionMode(self, mode):
        """Change the composition mode of the item (see QPainter::CompositionMode
        in the Qt documentation). This is useful when overlaying multiple ImageItems.

        ============================================  ============================================================
        **Most common arguments:**
        QtGui.QPainter.CompositionMode_SourceOver     Default; image replaces the background if it
                                                      is opaque. Otherwise, it uses the alpha channel to blend
                                                      the image with the background.
        QtGui.QPainter.CompositionMode_Overlay        The image color is mixed with the background color to
                                                      reflect the lightness or darkness of the background.
        QtGui.QPainter.CompositionMode_Plus           Both the alpha and color of the image and background pixels
                                                      are added together.
        QtGui.QPainter.CompositionMode_Multiply       The output is the image color multiplied by the background.
        ============================================  ============================================================
        """
        self.paintMode = mode
        self.update()

    def setBorder(self, b):
        self.border = fn.mkPen(b)
        self.update()

    def width(self):
        if self.image is None:
            return None
        axis = 0 if self.axisOrder == 'col-major' else 1
        return self.image.shape[axis]

    def height(self):
        if self.image is None:
            return None
        axis = 1 if self.axisOrder == 'col-major' else 0
        return self.image.shape[axis]

    def channels(self):
        if self.image is None:
            return None
        return self.image.shape[2] if self.image.ndim == 3 else 1

    def boundingRect(self):
        if self.image is None:
            return QtCore.QRectF(0., 0., 0., 0.)
        return QtCore.QRectF(0., 0., float(self.width()), float(self.height()))

    def setLevels(self, levels, update=True):
        """
        Set image scaling levels. Can be one of:

        * [blackLevel, whiteLevel]
        * [[minRed, maxRed], [minGreen, maxGreen], [minBlue, maxBlue]]

        Only the first format is compatible with lookup tables. See :func:`makeARGB <pyqtgraph.makeARGB>`
        for more details on how levels are applied.
        """
        if levels is not None:
            levels = np.asarray(levels)
        if not fn.eq(levels, self.levels):
            self.levels = levels
            self._effectiveLut = None
            if update:
                self.updateImage()

    def getLevels(self):
        return self.levels
        # return self.whiteLevel, self.blackLevel

    def setLookupTable(self, lut, update=True):
        """
        Set the lookup table (numpy array) to use for this image. (see
        :func:`makeARGB <pyqtgraph.makeARGB>` for more information on how this is used).
        Optionally, lut can be a callable that accepts the current image as an
        argument and returns the lookup table to use.

        Ordinarily, this table is supplied by a :class:`HistogramLUTItem <pyqtgraph.HistogramLUTItem>`
        or :class:`GradientEditorItem <pyqtgraph.GradientEditorItem>`.
        """
        if lut is not self.lut:
            self.lut = lut
            self._effectiveLut = None
            if update:
                self.updateImage()

    def setAutoDownsample(self, ads):
        """
        Set the automatic downsampling mode for this ImageItem.

        Added in version 0.9.9
        """
        self.autoDownsample = ads
        self.qimage = None
        self.update()

    def setOpts(self, update=True, **kargs):
        if 'axisOrder' in kargs:
            val = kargs['axisOrder']
            if val not in ('row-major', 'col-major'):
                raise ValueError('axisOrder must be either "row-major" or "col-major"')
            self.axisOrder = val
        if 'lut' in kargs:
            self.setLookupTable(kargs['lut'], update=update)
        if 'levels' in kargs:
            self.setLevels(kargs['levels'], update=update)
        # if 'clipLevel' in kargs:
        # self.setClipLevel(kargs['clipLevel'])
        if 'opacity' in kargs:
            self.setOpacity(kargs['opacity'])
        if 'compositionMode' in kargs:
            self.setCompositionMode(kargs['compositionMode'])
        if 'border' in kargs:
            self.setBorder(kargs['border'])
        if 'removable' in kargs:
            self.removable = kargs['removable']
            self.menu = None
        if 'autoDownsample' in kargs:
            self.setAutoDownsample(kargs['autoDownsample'])
        if update:
            self.update()

    def setRect(self, rect):
        """Scale and translate the image to fit within rect (must be a QRect or QRectF)."""
        self.resetTransform()
        self.translate(rect.left(), rect.top())
        self.scale(rect.width() / self.width(), rect.height() / self.height())

    def clear(self):
        self.image = None
        self.prepareGeometryChange()
        self.informViewBoundsChanged()
        self.update()

    def setImage(self, image=None, autoLevels=None, **kargs):
        r"""
        Update the image displayed by this item. For more information on how the image
        is processed before displaying, see :func:`makeARGB <pyqtgraph.makeARGB>`

        =================  =========================================================================
        **Arguments:**
        image              (numpy array) Specifies the image data. May be 2D (width, height) or
                           3D (width, height, RGBa). The array dtype must be integer or floating
                           point of any bit depth. For 3D arrays, the third dimension must
                           be of length 3 (RGB) or 4 (RGBA). See *notes* below.
        autoLevels         (bool) If True, this forces the image to automatically select
                           levels based on the maximum and minimum values in the data.
                           By default, this argument is true unless the levels argument is
                           given.
        lut                (numpy array) The color lookup table to use when displaying the image.
                           See :func:`setLookupTable <pyqtgraph.ImageItem.setLookupTable>`.
        levels             (min, max) The minimum and maximum values to use when rescaling the image
                           data. By default, this will be set to the minimum and maximum values
                           in the image. If the image array has dtype uint8, no rescaling is necessary.
        opacity            (float 0.0-1.0)
        compositionMode    See :func:`setCompositionMode <pyqtgraph.ImageItem.setCompositionMode>`
        border             Sets the pen used when drawing the image border. Default is None.
        autoDownsample     (bool) If True, the image is automatically downsampled to match the
                           screen resolution. This improves performance for large images and
                           reduces aliasing. If autoDownsample is not specified, then ImageItem will
                           choose whether to downsample the image based on its size.
        =================  =========================================================================


        **Notes:**

        For backward compatibility, image data is assumed to be in column-major order (column, row).
        However, most image data is stored in row-major order (row, column) and will need to be
        transposed before calling setImage()::

            imageitem.setImage(imagedata.T)

        This requirement can be changed by calling ``image.setOpts(axisOrder='row-major')`` or
        by changing the ``imageAxisOrder`` global configuration option apiref_config.


        """
        profile = debug.Profiler()

        gotNewData = False
        if image is None:
            if self.image is None:
                return
        else:
            gotNewData = True
            shapeChanged = (self.image is None or image.shape != self.image.shape)
            image = image.view(np.ndarray)
            if self.image is None or image.dtype != self.image.dtype:
                self._effectiveLut = None
            self.image = image
            if self.image.shape[0] > 2 ** 15 - 1 or self.image.shape[1] > 2 ** 15 - 1:
                if 'autoDownsample' not in kargs:
                    kargs['autoDownsample'] = True
            if shapeChanged:
                self.prepareGeometryChange()
                self.informViewBoundsChanged()

        profile()

        if autoLevels is None:
            if 'levels' in kargs:
                autoLevels = False
            else:
                autoLevels = True
        if autoLevels:
            img = self.image
            while img.size > 2 ** 16:
                img = img[::2, ::2]
            mn, mx = img.min(), img.max()
            if mn == mx:
                mn = 0
                mx = 255
            kargs['levels'] = [mn, mx]

        profile()

        self.setOpts(update=False, **kargs)

        profile()

        self.qimage = None
        self.update()

        profile()

        if gotNewData:
            self.sigImageChanged.emit()

    def dataTransform(self):
        """Return the transform that maps from this image's input array to its
        local coordinate system.

        This transform corrects for the transposition that occurs when image data
        is interpreted in row-major order.
        """
        # Might eventually need to account for downsampling / clipping here
        tr = QtGui.QTransform()
        if self.axisOrder == 'row-major':
            # transpose
            tr.scale(1, -1)
            tr.rotate(-90)
        return tr

    def inverseDataTransform(self):
        """Return the transform that maps from this image's local coordinate
        system to its input array.

        See dataTransform() for more information.
        """
        tr = QtGui.QTransform()
        if self.axisOrder == 'row-major':
            # transpose
            tr.scale(1, -1)
            tr.rotate(-90)
        return tr

    def mapToData(self, obj):
        tr = self.inverseDataTransform()
        return tr.map(obj)

    def mapFromData(self, obj):
        tr = self.dataTransform()
        return tr.map(obj)

    def quickMinMax(self, targetSize=1e6):
        """
        Estimate the min/max values of the image data by subsampling.
        """
        data = self.image
        while data.size > targetSize:
            ax = np.argmax(data.shape)
            sl = [slice(None)] * data.ndim
            sl[ax] = slice(None, None, 2)
            data = data[sl]
        return np.nanmin(data), np.nanmax(data)

    def updateImage(self, *args, **kargs):
        ## used for re-rendering qimage from self.image.

        ## can we make any assumptions here that speed things up?
        ## dtype, range, size are all the same?
        defaults = {
            'autoLevels': False,
        }
        defaults.update(kargs)
        return self.setImage(*args, **defaults)

    def render(self):
        # Convert data to QImage for display.

        profile = debug.Profiler()
        if self.image is None or self.image.size == 0:
            return

        # Request a lookup table if this image has only one channel
        if self.image.ndim == 2 or self.image.shape[2] == 1:
            if isinstance(self.lut, collections.Callable):
                lut = self.lut(self.image)
            else:
                lut = self.lut
        else:
            lut = None

        # Rick's Modification:  == 'never!'
        # if self.autoDownsample == 'never!':
        if True: #self.autoDownsample:
            # reduce dimensions of image based on screen resolution
            o = self.mapToDevice(QtCore.QPointF(0, 0))
            x = self.mapToDevice(QtCore.QPointF(1, 0))
            y = self.mapToDevice(QtCore.QPointF(0, 1))
            w = Point(x - o).length()
            h = Point(y - o).length()
            if w == 0 or h == 0:
                self.qimage = None
                return
            # self.xds = max(1, int(1.0 / w))
            # self.yds = max(1, int(1.0 / h))
            self.xds = int(1.0 / w)
            self.yds = int(1.0 / h)
            axes = [1, 0] if self.axisOrder == 'row-major' else [0, 1]
            # image = fn.downsample(self.image, xds, axis=axes[0])
            # image = fn.downsample(image, yds, axis=axes[1])
            image = self.image
            self._lastDownsample = (self.xds, self.yds)
        else:
            image = self.image

        # if the image data is a small int, then we can combine levels + lut
        # into a single lut for better performance
        levels = self.levels
        if levels is not None and levels.ndim == 1 and image.dtype in (np.ubyte, np.uint16):
            if self._effectiveLut is None:
                eflsize = 2 ** (image.itemsize * 8)
                ind = np.arange(eflsize)
                minlev, maxlev = levels
                levdiff = maxlev - minlev
                levdiff = 1 if levdiff == 0 else levdiff  # don't allow division by 0
                if lut is None:
                    efflut = fn.rescaleData(ind, scale=255. / levdiff,
                                            offset=minlev, dtype=np.ubyte)
                else:
                    lutdtype = np.min_scalar_type(lut.shape[0] - 1)
                    efflut = fn.rescaleData(ind, scale=(lut.shape[0] - 1) / levdiff,
                                            offset=minlev, dtype=lutdtype, clip=(0, lut.shape[0] - 1))
                    efflut = lut[efflut]

                self._effectiveLut = efflut
            lut = self._effectiveLut
            levels = None

        # Convert single-channel image to 2D array
        if image.ndim == 3 and image.shape[-1] == 1:
            image = image[..., 0]

        # Assume images are in column-major order for backward compatibility
        # (most images are in row-major order)
        if self.axisOrder == 'col-major':
            image = image.transpose((1, 0, 2)[:image.ndim])

        argb, alpha = fn.makeARGB(image, lut=lut, levels=levels)
        self.qimage = fn.makeQImage(argb, alpha, transpose=False)

    def paint(self, p, *args):
        profile = debug.Profiler()
        if self.image is None:
            return
        if self.qimage is None:
            self.render()
            if self.qimage is None:
                return
            profile('render QImage')
        if self.paintMode is not None:
            p.setCompositionMode(self.paintMode)
            profile('set comp mode')

        shape = self.image.shape[:2] if self.axisOrder == 'col-major' else self.image.shape[:2][::-1]
        # Rick's Modification: smoothScaled(1000, 1000):
        p.drawImage(QtCore.QRectF(0, 0, *shape), self.qimage.smoothScaled(shape[0]/self.xds, shape[1]/self.yds))
        # p.drawImage(QtCore.QRectF(0, 0, *shape), self.qimage)
        profile('p.drawImage')
        if self.border is not None:
            p.setPen(self.border)
            p.drawRect(self.boundingRect())

    def save(self, fileName, *args):
        """Save this image to file. Note that this saves the visible image (after scale/color changes), not the original data."""
        if self.qimage is None:
            self.render()
        self.qimage.save(fileName, *args)

    def getHistogram(self, bins='auto', step='auto', perChannel=False, targetImageSize=200,
                     targetHistogramSize=500, **kwds):
        """Returns x and y arrays containing the histogram values for the current image.
        For an explanation of the return format, see numpy.histogram().

        The *step* argument causes pixels to be skipped when computing the histogram to save time.
        If *step* is 'auto', then a step is chosen such that the analyzed data has
        dimensions roughly *targetImageSize* for each axis.

        The *bins* argument and any extra keyword arguments are passed to
        np.histogram(). If *bins* is 'auto', then a bin number is automatically
        chosen based on the image characteristics:

        * Integer images will have approximately *targetHistogramSize* bins,
          with each bin having an integer width.
        * All other types will have *targetHistogramSize* bins.

        If *perChannel* is True, then the histogram is computed once per channel
        and the output is a list of the results.

        This method is also used when automatically computing levels.
        """
        if self.image is None:
            return None, None
        if step == 'auto':
            step = (int(np.ceil(self.image.shape[0] / targetImageSize)),
                    int(np.ceil(self.image.shape[1] / targetImageSize)))
        if np.isscalar(step):
            step = (step, step)
        stepData = self.image[::step[0], ::step[1]]

        if bins == 'auto':
            mn = stepData.min()
            mx = stepData.max()
            if stepData.dtype.kind in "ui":
                # For integer data, we select the bins carefully to avoid aliasing
                step = np.ceil((mx - mn) / 500.)
                bins = np.arange(mn, mx + 1.01 * step, step, dtype=np.int)
            else:
                # for float data, let numpy select the bins.
                bins = np.linspace(mn, mx, 500)

            if len(bins) == 0:
                bins = [mn, mx]

        kwds['bins'] = bins

        if perChannel:
            hist = []
            for i in range(stepData.shape[-1]):
                stepChan = stepData[..., i]
                stepChan = stepChan[np.isfinite(stepChan)]
                h = np.histogram(stepChan, **kwds)
                hist.append((h[1][:-1], h[0]))
            return hist
        else:
            stepData = stepData[np.isfinite(stepData)]
            hist = np.histogram(stepData, **kwds)
            return hist[1][:-1], hist[0]

    def setPxMode(self, b):
        """
        Set whether the item ignores transformations and draws directly to screen pixels.
        If True, the item will not inherit any scale or rotation transformations from its
        parent items, but its position will be transformed as usual.
        (see GraphicsItem::ItemIgnoresTransformations in the Qt documentation)
        """
        self.setFlag(self.ItemIgnoresTransformations, b)

    def setScaledMode(self):
        self.setPxMode(False)

    def getPixmap(self):
        if self.qimage is None:
            self.render()
            if self.qimage is None:
                return None
        return QtGui.QPixmap.fromImage(self.qimage)

    def pixelSize(self):
        """return scene-size of a single pixel in the image"""
        br = self.sceneBoundingRect()
        if self.image is None:
            return 1, 1
        return br.width() / self.width(), br.height() / self.height()

    def viewTransformChanged(self):
        if self.autoDownsample:
            self.qimage = None
            self.update()

    def mouseDragEvent(self, ev):
        if ev.button() != QtCore.Qt.LeftButton:
            ev.ignore()
            return
        elif self.drawKernel is not None:
            ev.accept()
            self.drawAt(ev.pos(), ev)

    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.RightButton:
            if self.raiseContextMenu(ev):
                ev.accept()
        if self.drawKernel is not None and ev.button() == QtCore.Qt.LeftButton:
            self.drawAt(ev.pos(), ev)

    def raiseContextMenu(self, ev):
        menu = self.getMenu()
        if menu is None:
            return False
        menu = self.scene().addParentContextMenus(self, menu, ev)
        pos = ev.screenPos()
        menu.popup(QtCore.QPoint(pos.x(), pos.y()))
        return True

    def getMenu(self):
        if self.menu is None:
            if not self.removable:
                return None
            self.menu = QtGui.QMenu()
            self.menu.setTitle("Image")
            remAct = QtGui.QAction("Remove image", self.menu)
            remAct.triggered.connect(self.removeClicked)
            self.menu.addAction(remAct)
            self.menu.remAct = remAct
        return self.menu

    def hoverEvent(self, ev):
        if not ev.isExit() and self.drawKernel is not None and ev.acceptDrags(QtCore.Qt.LeftButton):
            ev.acceptClicks(QtCore.Qt.LeftButton)  ## we don't use the click, but we also don't want anyone else to use it.
            ev.acceptClicks(QtCore.Qt.RightButton)
        elif not ev.isExit() and self.removable:
            ev.acceptClicks(QtCore.Qt.RightButton)  ## accept context menu clicks

    def tabletEvent(self, ev):
        pass
        # print(ev.device())
        # print(ev.pointerType())
        # print(ev.pressure())

    def drawAt(self, pos, ev=None):
        pos = [int(pos.x()), int(pos.y())]
        dk = self.drawKernel
        kc = self.drawKernelCenter
        sx = [0, dk.shape[0]]
        sy = [0, dk.shape[1]]
        tx = [pos[0] - kc[0], pos[0] - kc[0] + dk.shape[0]]
        ty = [pos[1] - kc[1], pos[1] - kc[1] + dk.shape[1]]

        for i in [0, 1]:
            dx1 = -min(0, tx[i])
            dx2 = min(0, self.image.shape[0] - tx[i])
            tx[i] += dx1 + dx2
            sx[i] += dx1 + dx2

            dy1 = -min(0, ty[i])
            dy2 = min(0, self.image.shape[1] - ty[i])
            ty[i] += dy1 + dy2
            sy[i] += dy1 + dy2

        ts = (slice(tx[0], tx[1]), slice(ty[0], ty[1]))
        ss = (slice(sx[0], sx[1]), slice(sy[0], sy[1]))
        mask = self.drawMask
        src = dk

        if isinstance(self.drawMode, collections.Callable):
            self.drawMode(dk, self.image, mask, ss, ts, ev)
        else:
            src = src[ss]
            if self.drawMode == 'set':
                if mask is not None:
                    mask = mask[ss]
                    self.image[ts] = self.image[ts] * (1 - mask) + src * mask
                else:
                    self.image[ts] = src
            elif self.drawMode == 'add':
                self.image[ts] += src
            else:
                raise Exception("Unknown draw mode '%s'" % self.drawMode)
            self.updateImage()
    def setDrawKernel(self, kernel=None, mask=None, center=(0, 0), mode='set'):
        self.drawKernel = kernel
        self.drawKernelCenter = center
        self.drawMode = mode
        self.drawMask = mask

    def removeClicked(self):
        ## Send remove event only after we have exited the menu event handler
        self.removeTimer = QtCore.QTimer()
        self.removeTimer.timeout.connect(self.emitRemoveRequested)
        self.removeTimer.start(0)

    def emitRemoveRequested(self):
        self.removeTimer.timeout.disconnect(self.emitRemoveRequested)
        self.sigRemoveRequested.emit(self)


