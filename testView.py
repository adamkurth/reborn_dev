#!/usr/bin/env python

import pad
import convert
import data
import imview
from PyQt4 import QtGui
import sys





#p = pad.panelArray()
#p.loadPypadTxt("ex.txt")

#for panel in p.panels:
#    print(panel.T)
#    print(panel.pixSize)

p = convert.crystfelToPanelArray("examples/example1.geom")

print(p)


#print(p.panels[0].dataPlan.dataField)
#print(p)
#print(p.panels[63].I)


# im = p.panels[0].I
# app = QtGui.QApplication([])
# ex = imview.simple2DView()
# ex.loadImage(im)
# sys.exit(app.exec_())
