import numpy as np
import matplotlib.pyplot as plt
from bornagain import detector, dataio

filePath = 'run178730.hdf5'

reader = dataio.saclaReader(filePath)
pa = detector.PanelList()

frameNumber = 3
reader.getFrame(pa, frameNumber)
im = pa.assembledData

plt.imshow(im, interpolation='nearest', cmap='gray')
plt.show()
