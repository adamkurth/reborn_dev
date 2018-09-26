import weakref
import numpy as np
import pyqtgraph as pg


def keep_open():
    r"""
    Simple helper that keeps qtgraph window open when you run a script from the terminal.
    """

    pg.QtGui.QApplication.instance().exec_()


class MultiHistogramLUTWidget(pg.GraphicsView):

    r"""
    This is the equivalent of :class:`pyqtgraph.HistogramLUTWidget`, but wraps
    :class:`MultiHistogramLUTWidget <bornagain.external.pyqtgraph.MultiHistogramLUTItem>` instead of
    :class:`pyqtgraph.HistogramLUTItem`.
    """

    def __init__(self, parent=None, *args, **kargs):
        background = kargs.get('background', 'default')
        pg.GraphicsView.__init__(
            self,
            parent,
            useOpenGL=False,
            background=background)
        self.item = MultiHistogramLUTItem(*args, **kargs)
        self.setCentralItem(self.item)
        self.setSizePolicy(
            pg.QtGui.QSizePolicy.Preferred,
            pg.QtGui.QSizePolicy.Expanding)
        self.setMinimumWidth(95)

    def sizeHint(self):
        r"""
        Undocumented pyqtgraph method.
        """

        return pg.QtCore.QSize(115, 200)

    def __getattr__(self, attr):
        return getattr(self.item, attr)


class MultiHistogramLUTItem(pg.HistogramLUTItem):

    r"""
    This is a bornagain extension to the :class:`pyqtgraph.HistogramLUTItem` that allows control
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

        phonyarray = np.ravel([im.image for im in imgs])
        phonyarray = phonyarray[0:(len(phonyarray) - (len(phonyarray) % 2))]
        phonyarray = phonyarray.reshape([2, int(len(phonyarray) / 2)])

        self.imageItemStrong = pg.ImageItem(phonyarray)
        self.imageItem = weakref.ref(self.imageItemStrong)  # TODO: fix this up
        self.regionChanged()
        self.imageChanged(autoLevel=True)

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
