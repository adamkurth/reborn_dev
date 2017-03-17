""" 
Totally experimental.  Nothing useful here.
"""

import bornagain
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui

# create GUI
app = QtGui.QApplication([])
w = pg.GraphicsWindow(size=(800, 800))
w.setWindowTitle('QTV')

w4 = w.addLayout()
v4 = w4.addViewBox(row=1, col=0, lockAspect=True)
g = pg.GridItem()
v4.addItem(g)

rs = []
imgs = []

# for i in range(1):
# 
#     pan = p[i]
# 
#     xmn = pan.T[0] / pan.pixSize
#     ymn = pan.T[1] / pan.pixSize
# 
#     xmx = xmn + pan.nF
#     ymx = ymn + pan.nS
# 
#     bb = pan.getRealSpaceBoundingBox() / pan.pixSize
# 
#     rs.append(pg.ROI([xmn, xmx], [ymn, ymx]))
#     rs[i].addRotateHandle([1, 0], [0.5, 0.5])
#     imgs.append(pg.ImageItem(pan.data))
#     imgs[i].setParentItem(rs[i])
#     v4.addItem(rs[i])
# 
# 
# v4.disableAutoRange('xy')
# v4.autoRange()
# 
# QtGui.QApplication.instance().exec_()


# p = pad.panelArray()
# p.loadPypadTxt("ex.txt")

# for panel in p.panels:
#    print(panel.T)
#    print(panel.pixSize)


# print(p)


# print(p.panels[0].dataPlan.dataField)
# print(p)
# print(p.panels[63].I)


# im = p.panels[0].I
# app = QtGui.QApplication([])
# ex = imview.simple2DView()
# ex.loadImage(im)
# sys.exit(app.exec_())
