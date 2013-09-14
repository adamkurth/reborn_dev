'''
Created on Jul 27, 2013

@author: kirian
'''

from PyQt4 import QtGui
import pyqtgraph as pg
import numpy as np

class simple2DView(QtGui.QMainWindow):

    def __init__(self):

        super(simple2DView, self).__init__()

        self.initUI()

    def initUI(self):

        self.resize(800, 800)

        exitAction = QtGui.QAction(QtGui.QIcon('exit.png'), '&Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(QtGui.qApp.quit)

        openFile = QtGui.QAction(QtGui.QIcon('open.png'), '&Open', self)
        openFile.setShortcut('Ctrl+O')
        openFile.setStatusTip('Open image file')
        openFile.triggered.connect(self.showDialog)

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(exitAction)
        fileMenu.addAction(openFile)

    def center(self):
        qr = self.frameGeometry()
        cp = QtGui.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def loadImage(self, im):

        imv = pg.ImageView()
        self.setCentralWidget(imv)
        self.show()
        self.setWindowTitle("window title")
        imv.setImage(im)
        mx = np.amax(im)
        imv.setLevels(0, mx)

    def showDialog(self):

        fname = QtGui.QFileDialog.getOpenFileName(self, 'Open data file', '$HOME')




class simple3DView(QtGui.QMainWindow):

    def __init__(self):

        super(simple3DView, self).__init__()

        self.initUI()

    def initUI(self):

        self.resize(800, 800)

        exitAction = QtGui.QAction(QtGui.QIcon('exit.png'), '&Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(QtGui.qApp.quit)

        openFile = QtGui.QAction(QtGui.QIcon('open.png'), '&Open', self)
        openFile.setShortcut('Ctrl+O')
        openFile.setStatusTip('Open image file')
        openFile.triggered.connect(self.showDialog)

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(exitAction)
        fileMenu.addAction(openFile)

    def center(self):
        qr = self.frameGeometry()
        cp = QtGui.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def loadImage(self, im):

        imv = pg.ImageView()
        self.setCentralWidget(imv)
        self.show()
        self.setWindowTitle("window title")
        imv.setImage(im)
        mx = np.amax(im)
        imv.setLevels(0, mx)

    def showDialog(self):

        fname = QtGui.QFileDialog.getOpenFileName(self, 'Open data file', '$HOME')



# class simple3DView(QtGui.QMainWindow):

#    pass  # pg.opengl.GLImageItem(data, smooth=False)


