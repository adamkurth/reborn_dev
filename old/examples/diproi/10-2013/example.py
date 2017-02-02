import numpy as np
import matplotlib.pyplot as plt
from bornagain import dataio, detector

# The data reader - a super class of panelList with helpers for loading data from files
pa = dataio.diproiReader()
# Set the list of files that we wish to read
pa.fileList = ['A13_IMG_53908109.h5', 'A13_IMG_53908329.h5']
# Set the data field, which is where the intensity data are
pa.dataField = 'image/ccd1'
# Load data from a particular frame
pa.getFrame(1)
# Set up the geometry
pa.wavelength = 32e-9
pa[0].T = np.array([-1024, -1024, 0.05])  # Translation to first pixel
pa[0].F = np.array([1, 0, 0]) * 50e-6  # Fast-scan direction (and pixel size)
pa[0].S = np.array([0, 1, 0]) * 50e-6  # Slow-scan direction (and pixel size)
# Check geometry
print(pa.checkGeometry())
print(pa)

# Assemble the data for viewing
im = pa.assembledData
# View the data
plt.imshow(im, interpolation='nearest', cmap='gray')
plt.show()

