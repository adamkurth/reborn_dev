"""
Basically everything we need for live image analysis

"""

import sys
import pyqtgraph as pg

import numpy as np

import pyqtgraph as      pg
from   pyqtgraph import  GraphicsView
from   pyqtgraph.Qt import QtCore, QtGui
# from   pyqtgraph import QtCore, QtGui, GraphicsView
# from PyQt4 import QtCore, QtGui

##########################################
## Global methods
##########################################

def make_known_image(shape,sigma=0): 

    arr   = np.ones(shape, dtype=float)
    l,w   = int(shape[0]/2) , int(shape[1]/2)
    l2,w2 = int(l/2), int(w/2)

    arr[l-l2, w   ] += 10
    arr[l+l2, w-w2] += 10
    arr[l+l2, w+w2] += 10

    if sigma > 0: 
        arr = pg.gaussianFilter(arr, sigma)

    return arr

def make_random_image(shape, sigma=3):

    imageData = np.random.normal(size=shape)
    imageData[10:20, 30:40] += 2.
    if sigma > 0:
        imageData = pg.gaussianFilter(imageData, sigma)
    imageData += 0.1* np.random.normal(size=shape)

    return imageData

def make_roi():

    # Custom ROI for selecting an image region
    
    """ 
        http://www.pyqtgraph.org/documentation/graphicsItems/roi.html
        
        pg.ROI.addScaleHandle:

            pos     (length-2 sequence) The position of the handle relative to the shape of the ROI. A value of (0,0) indicates the origin, whereas (1, 1) indicates the upper-right corner, regardless of the ROI’s size.
            center  (length-2 sequence) The center point around which rotation takes place.
            item    The Handle instance to add. If None, a new handle will be created.
            name    (optional). Handles are identified by name when calling getLocalHandlePositions and getSceneHandlePositions.            
    """    

    roi = pg.ROI(pos=[0,0], size=[10, 10])
    roi.addScaleHandle(pos=[0.5, 1], center=[0.5, 0.5])  
    roi.addScaleHandle(pos=[0, 0.5], center=[0.5, 0.5])
    roi.setZValue(10)  # draw ROI above image

    return roi

def make_isocurve(parent):
    
    iso = pg.IsocurveItem(level=0.8, pen='g')
    iso.setParentItem(parent)
    iso.setZValue(5)   

    return iso   

def make_isoline():

    isoLine = pg.InfiniteLine(angle=0, movable=True, pen='g')
    
    return isoLine

def make_hist(image, isoline = None):

    hist = pg.HistogramLUTItem()
    hist.setImageItem(image)
    hist.vb.setMouseEnabled(y=False)    # makes user interaction a little easier

    if isoline: 
        isoline.setValue(0.2)
        isoline.setZValue(1000)     # bring isoline above contrast controls
        hist.vb.addItem(isoline)

    return hist    

##########################################
## Classes 
##########################################

class ImageView():

    def __init__(self,
                 debug     = False,
                 randomImg = False,
                 hasROI    = False,
                 hasIso    = False,
                 hasHist   = True,
                 hasLine   = False,
                 imageData = None, 
                 lineData  = np.array([]),
                 resolution= (100, 100),
                 ):
    
        ##############################
        ## Make Window

        # pg.setConfigOptions(imageAxisOrder='row-major')

        ## Main App/Widget
        self.app = pg.mkQApp()      # must be "self"
        win = pg.GraphicsLayoutWidget()
        win.resize(1200, 800)
        win.setWindowTitle('Image Analysis')

        ## Item for displaying image data
        img = pg.ImageItem(border="w")   

        ## Image view (ViewBox + axes)
        self.p1  = win.addPlot()   
        self.p1.addItem(img)

        win.show()

        self.win = win
        self.img = img

        ##############################
        
        if randomImg:
            make_dummy_image = make_random_image
        else:
            make_dummy_image = make_known_image

        if debug:
            print("\n***** Debug Mode *****\n")

            imageData = make_dummy_image(shape=resolution)
            if hasLine:
                lineData  = np.random.randint(low=0 , high=np.min(resolution), size=(5,2))
            else: 
                lineData  = None
            
            hasROI  = True
            hasIso  = True
            hasHist = True
        
        if imageData is not None:
            print("Using Dummy Image Data\n")
            
            imageData = make_dummy_image(shape=resolution)

        self.lineData = lineData
        self.imageData= imageData
        self.img.setImage(imageData)  
        # self.img.scale(0.2, 0.2)        # scale img
        # self.img.translate(-50, 0)      # move img

        if lineData is not None: 
            self.p1.plot(lineData)

        self.p1.autoRange()             # zoom to fit image

        self.roi     = None
        self.hist    = None
        self.isoLine = None
        self.isoCurve= None
    
        if hasIso:

            self.isoCurve = make_isocurve(parent=self.img)
            self.isoCurve.setData(pg.gaussianFilter(imageData, sigma=2))  # smoothed imageData

            self.isoLine  = make_isoline() 
            self.isoLine.sigDragged.connect(self.updateIsocurve)
            self.updateIsocurve()

        if hasHist:
            self.hist = make_hist(image=self.img, isoline=self.isoLine)
            win.addItem(self.hist)  
            self.hist.setLevels(imageData.min(), imageData.max())  

        if hasROI:
            self.roi = make_roi()
            self.p1.addItem(self.roi)

            ## ROI plots
            win.nextRow()
            self.roiAvg_axis0 = win.addPlot(title="ROI Avg (Axis 0)",colspan=2)
            self.roiAvg_axis0.setMaximumHeight(250)

            win.nextRow()
            self.roiAvg_axis1 = win.addPlot(title="ROI Avg (Axis 1)", colspan=2)
            self.roiAvg_axis1.setMaximumHeight(250)
            
            ## ROI Update
            self.roi.sigRegionChanged.connect(self.updateROI)
            self.updateROI()
        
        # self.status = app.exec_()

    def updateImage(self):
        
        pass

    def updateROI(self):  # handle user interaction

        selected = self.roi.getArrayRegion(self.imageData, self.img)
        self.roiAvg_axis0.plot(selected.mean(axis=0), clear=True) 
        self.roiAvg_axis1.plot(selected.mean(axis=1), clear=True)   

    def updateIsocurve(self):
        self.isoCurve.setLevel(self.isoLine.value())       


class SlideShow():

    def __init__(self,
                 debug     = False,
                 nFrames   = 10,
                 randomImg = True,
                 hasROI    = False,
                 hasIso    = False,
                 hasHist   = True,
                 hasLine   = False,
                 imageData = None, 
                 lineData  = np.array([]),
                 resolution= (100, 100),
                 ):

        ##############################
        ## Make Window

        # pg.setConfigOptions(imageAxisOrder='row-major')

        ## Main App/Widget
        self.app = pg.mkQApp()      # Must be "self"
        win = QtGui.QMainWindow()
        win.resize(1200, 800)
        win.setWindowTitle('Slide Show')

        imv = pg.ImageView()
        win.setCentralWidget(imv)

        win.show()
        
        self.win = win  # Must be done, or else win won't show!!! Don't know why

        ##############################
     
        frameStack = make_random_image(shape=(nFrames,*resolution))
        timeline   = np.linspace(start=0., stop=1. , num=nFrames)
        imv.setImage(frameStack, xvals=timeline)  

        # self.status = app.exec_()


class ImageViewer1():  # Use pg.GraphicsLayoutWidget,  pg.ImageItem,  pg.addPlot (i.e. ViewBox + axes)

    def __init__(self):

        self.app    = pg.mkQApp()   
        self.update = pg.QtGui.QApplication.processEvents
        self.win    = pg.GraphicsLayoutWidget()
        self.img    = pg.ImageItem()        # Item for displaying image data
        self.p      = self.win.addPlot()    # plot area (ViewBox + axes) for the image
        
        self.win.resize(800, 800)
        self.img.setImage(image, autoLevels=True, autoRange=True)
        self.p.addItem(img)

        self.win.show()
        
    def set_image(self, image, pause = False, clear = True, autoLevels = True):
        # self.win.show()
        if clear: self.p.clear()
        self.img.setImage(image, autoLevels=autoLevels, autoRange=True) 
        # self.img.autoRange()
        self.update()

        if pause: input("Press Enter to continue...")


class ImageViewer2():  # Use pg.PlotItem (i.e. "axis") and  pg.ImageView (i.e. "window")

    def __init__(self):

        self.app    = pg.mkQApp()   
        self.update = pg.QtGui.QApplication.processEvents
        self.imAx   = pg.PlotItem()
        self.imWin  = pg.ImageView(view = self.imAx)

    def set_image(self, image, pause = False, clear = True, autoLevels = True):
        self.imWin.show()
        if clear: self.imAx.clearPlots()
        self.imWin.setImage(image, autoLevels = autoLevels)
        self.imWin.autoRange()
        self.update()

        if pause: input("Press Enter to continue...")     # In python3 it's just input()

    def plot_data(self, data, pause = False, clear = True, autoLevels = True, thickness = 5, color = 'r', connectPoints=True):
        
        self.imWin.show()
        
        if clear: self.imAx.clearPlots()
        if connectPoints: 
            pen = pg.mkPen(color, width=thickness)
            self.imAx.plot(*data, autoLevels=autoLevels, pen = pen)
        else:
            self.imAx.plot(*data, autoLevels=autoLevels, pen = None, symbolBrush=color)

        self.update()
        
        if pause: input("Press Enter to continue...")     # In python3 it's just input()

    def execute(self, verbose = False):
        
        if verbose: print("self.execute():")
        
        self.imWin.show()  
        self.update()  

        if verbose: 
            print("\t--> show()")
            print("\t--> update()")

        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            # APP = QtGui.QApplication.instance()   # here, same as self.app
            if verbose: print("\t--> self.app:", self.app)
            if self.app is not None: 
                if verbose: print("\t--> self.app.exec_()")
                self.app.exec_() 

        if verbose: print("\t--> QtGui App Terminated")

##########################################

def main(image):
    viewer = ImageViewer2()
    viewer.set_image(image)
    # QtGui.QApplication.instance().exec_()

##########################################


if __name__ == '__main__':

    # main()

    s=SlideShow()   
    v=ImageView(debug=True, randomImg=False) 

    # pg.exit()

    ## Start Qt event loop unless running in interactive mode or using pyside.
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        
        APP = QtGui.QApplication.instance()
        if APP is not None: 
            APP.exec_() 

    pg.exit()  # MUST! Avoids many pyqt(graph) seg. faults and bugs. ref: http://www.pyqtgraph.org/documentation/functions.html#pyqtgraph.exit
