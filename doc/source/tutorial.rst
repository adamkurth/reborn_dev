Tutorial
********


Load some panel information from a CrystFEL geom file::

    from pydiffract import convert
    
    [pa, reader] = convert.crystfelToPanelList("examples/example1.geom")
    reader.getShot(pa, "examples/example1.h5")

Load a SACLA hdf5 file::

	import h5py
	import numpy as np
	import matplotlib.pyplot as plt
	from pydiffract import detector, dataio
	
	filePath = '178790_10.h5'
	filePath = 'run178730.hdf5'
	
	reader = dataio.saclaReader(filePath)
	pa = detector.panelList()
	reader.getFrame(pa,0)
	
	im = pa.assembledData
	
	plt.imshow(im, interpolation='none')
	plt.show()
	
