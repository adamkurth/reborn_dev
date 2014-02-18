import numpy as np
import matplotlib.pyplot as plt
from pydiffract import dataio, detector

# The data reader - a super class of panelList with helpers for loading data from files
pa = dataio.diproiReader()
# Set the list of files that we wish to read
pa.fileList = ['A13_IMG_53908109.h5', 'A13_IMG_53908329.h5']
# Set the data field
pa.dataField = 'image/ccd1'
# Set some geometry
pa.wavelength = 32e-9
# Load data from a particular frame
pa.getFrame(1)
# Assemble the data for viewing
im = pa.assembledData
# View the data
plt.imshow(im, interpolation='none', cmap='gray')
plt.show()