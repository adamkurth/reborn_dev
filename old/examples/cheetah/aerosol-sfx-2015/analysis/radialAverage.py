#!/usr/bin/env python

from bornagain import dataio, utils
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

getter = dataio.frameGetter()
getter.reader = dataio.cheetahH5Reader()
getter.loadCrystfelGeometry(geomFile="./best.geom")
getter.fileList = glob('../hdf5/r0009-test2/data*/LCLS*.h5') #["../hdf5/r0048-gv2/data1/LCLS_2015_Jun26_r0048_143302_11f13.h5"]
getter.getFrame(0)

y = getter.reader.panelList.radialProfile()
# plt.plot(y)
# plt.show()

n = getter.nFrames
m = len(y)

stack=np.zeros([n,m])

for i in np.arange(0,n):
	print(i)
	pl = getter.getFrame(i)
	stack[i,:] = pl.radialProfile()

plt.imshow(stack)
plt.show()
