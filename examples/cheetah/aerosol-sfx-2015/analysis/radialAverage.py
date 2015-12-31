#!/usr/bin/env python

from pydiffract import dataio, utils
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

getter = dataio.frameGetter()
getter.reader = dataio.cheetahH5Reader()
getter.loadCrystfelGeometry(geomFile="./best.geom")
getter.fileList = glob('../hdf5/r0012-test2/data*/LCLS*.h5') #["../hdf5/r0048-gv2/data1/LCLS_2015_Jun26_r0048_143302_11f13.h5"]
print(getter.fileList)
getter.getFrame(0)

y = getter.reader.panelList.radialProfile()
#plt.plot(y)
#plt.show()

n = 1 #pl.nFrames
m = len(y)

stack=np.zeros([n,m])

for i in np.arange(0,n):
	print(i)
	getter.getFrame(i)
	stack[i,:] = getter.reader.panelList.radialProfile()

plt.imshow(stack)
plt.show()

