import sys
import pyqtgraph as pg
import numpy as np
from pyqtgraph.Qt import QtGui, QtCore
sys.path.append('..')
import bornagain as ba

# pg.setConfigOptions(crashWarning=False,imageAxisOrder='row-major')



###################################################################
# First generate some pixel-array data.
####################################################################

distance = 0.1
pixel_size = 100e-6
n_pixels = 1000

pad = ba.detector.PADGeometry()
pad.simple_setup(n_pixels=n_pixels, pixel_size=pixel_size, distance=distance)

data = np.random.random(pad.shape())
data[0,0:5] = 2  # This is to indicate the first pixel in memory and the fast-scan direction
for i in range(0,10):
    data[2**i,:] = 2
    data[:,2**i] = 2




###################################################################
# Now we will attempt to make a viewer.
####################################################################

class PADView(QtGui.QMainWindow):

    def __init__(self, pad, data):

        super(PADView, self).__init__()
        self.phony_distance = 0.1
        self.phony_pixel_size = 0.001
        self.pad = pad
        self.data = data
        self.initUI()
        self.glw = pg.GraphicsLayoutWidget()
        self.pltitem = self.glw.addPlot()
        self.pltitem.getViewBox().setAspectLocked(True)
        self.pltitem.getViewBox().invertX(True)
        self.hist = pg.HistogramLUTItem()
        self.glw.addItem(self.hist)
        self.imgitems = []
        self.append_pads(pad, data)
        self.pltitem.addItem(pg.InfiniteLine(0,angle=0,pen='r'))
        self.pltitem.addItem(pg.InfiniteLine(0,angle=90,pen='r'))
        self.setCentralWidget(self.glw)
        self.setWindowState(self.windowState() & ~QtCore.Qt.WindowMinimized | QtCore.Qt.WindowActive)
        self.activateWindow()
        self.show()

    def initUI(self):

        self.setWindowTitle('gwiz!')
        self.resize(1000, 800)
        self.statusBar()
        # self.center()

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
        # showScan.triggered.connect(self.showFastScans)
        viewMenu.addAction(showScan)

        # framePanels = QtGui.QAction(QtGui.QIcon('open.png'), 'Panel frames', self)
        # framePanels.setStatusTip('Draw frames around panels')
        # framePanels.triggered.connect(self.framePanels)
        # viewMenu.addAction(framePanels)

        # showAxes = QtGui.QAction(QtGui.QIcon('open.png'), 'Axes', self)
        # showAxes.setStatusTip('Draw axes (red,green,blue = x,y,z)')
        # showAxes.triggered.connect(self.showAxes)
        # viewMenu.addAction(showAxes)

    def showOpenFileDialog(self):

        fname = QtGui.QFileDialog.getOpenFileName(self, 'Open file', '$HOME')
        print('Not doing anything with %s' % fname)
        # self.loadImage(fname)

    def append_pads(self, pad, data):

        if not isinstance(pad, list):
            pad = [pad]
            dat = [data]

        for (p,d) in zip(pad,data):
            psiz = p.pixel_size()
            scale = psiz
            imgitem = pg.ImageItem()
            imgitem.setAutoDownsample(True)
            imgitem.setImage(d.T)
            imgitem.scale(scale, scale)
            imgitem.translate(-0.5*psiz, -0.5*psiz)
            # imgitem.rotate(20)
            print('pixel size', p.pixel_size())
            #/self.phony_pixel_size*self.phony_distance/p.t_vec.flat[2]
            print('t_vec', p.t_vec)
            # imgitem.translate(p.t_vec.flat[1],p.t_vec.flat[0])
            # print(imgitem.dataTransform())
            self.imgitems.append(imgitem)
            self.pltitem.addItem(imgitem)
            self.hist.setImageItem(imgitem)




qtguiapp = QtGui.QApplication([])
padview = PADView([pad], [data])
qtguiapp.exec_()









# Based on the pyqtgraph examples, I started with the GraphicsLayoutWidget class, which seemed useful.  More info can
# be found here:
# http://www.pyqtgraph.org/documentation/plotting.html#organization-of-plotting-classes
# The problem with GraphicsLayoutWidget is that it is a couple of lines long, and therefore does practically nothing
# except for keeping me ignorant of how QT and pyqtgraph actually work.  Since pyqtgraph documentation is incomplete
# in so many ways, I've found it necessary to read the code.
# glw = pg.GraphicsLayoutWidget()
# glw.setWindowTitle('gwiz!')
# glw.resize(1000, 800)

# Questions:
# == What's the difference between GraphicsItem, GraphicsView, GraphicsWidget?
# ---- GraphicsItem inherits from Object.  It is rather complicated and beyond my understanding of QT (and Python).
# ---- GraphicsView inherits from QGraphicsView.  See documentation.  Removes scrollbars etc.
# ---- GraphicsWidget inherits from GraphicsItem and QtGui.QGraphicsWidget
# Some notes from QT documentation:
# The QGraphicsWidget class is the base class for all widget items in a QGraphicsScene. QGraphicsWidget is an extended
# base item that provides extra functionality over QGraphicsItem.  Unlike QGraphicsItem, QGraphicsWidget is not an
# abstract class; you can create instances of a QGraphicsWidget without having to subclass it.  QGraphicsWidget can be
# used as a base item for your own custom item if you require advanced input focus handling, e.g., tab focus and
# activation, or layouts.




#         self._PanelList = None
#         self.panelItems = None



# Hopefully this approach is closer to the base classes.  Still, I don't see anything that exposes the creation of
# the main QT window.
# gv = pg.GraphicsView()
# gl = pg.GraphicsLayout()
# gv.setCentralItem(gl)
# gv.setWindowTitle('gwiz!')
# gv.resize(1000, 800)


# The plot item below is hard to understand.
# pltitem = gl.addPlot()
# pltitem.getViewBox().setAspectLocked(True)
#
#
# imgitem = pg.ImageItem()
# imgitem.setImage(data)
#
# pltitem.addItem(imgitem)

# imgitem.scale(0.2, 0.2)
# imgitem.translate(-50, 0)
#
# hist = pg.HistogramLUTItem()
# hist.setImageItem(imgitem)
#
# gl.addItem(hist)


# gv.show()




# QtGui.QApplication.instance().exec_()