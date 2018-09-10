import sys
sys.path.append("../..")
import numpy as np
import bornagain as ba
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui

import bornagain.simulate.clcore as clcore


pl = ba.detector.PanelList()
p = ba.detector.Panel()
p.simple_setup(100,150,100e-6,0.1,1.5e-6,T=[10000e-6,10000e-6,0])
pl.append(p)
q = pl.Q

N = 2
x = np.arange(0,N)*10e-10
[xx,yy,zz] = np.meshgrid(x,x,x,indexing='ij')
r = np.zeros([N**3,3])
r[:,0] = zz.flatten()
r[:,1] = yy.flatten()
r[:,2] = xx.flatten()

f = np.ones(r.shape[0])

A = clcore.phase_factor_qrf(q,r,f)
I = np.abs(A)**1

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

for i in range(1):

    pan = pl[i]

    xmn = pan.T[0] / pan.pixel_size
    ymn = pan.T[1] / pan.pixel_size

    xmx = xmn + pan.nF
    ymx = ymn + pan.nS

    bb = pan.getRealSpaceBoundingBox() / pan.pixel_size

    rs.append(pg.ROI([xmn, xmx], [ymn, ymx]))
    rs[i].addRotateHandle([1, 0], [0.5, 0.5])
    imgs.append(pg.ImageItem(pan.data))
    imgs[i].setParentItem(rs[i])
    v4.addItem(rs[i])


v4.disableAutoRange('xy')
v4.autoRange()

QtGui.QApplication.instance().exec_()


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
