PAD data and geometry formats
=============================

The most common diffraction detector is a Pixel-Array Detector (PAD).  The basic concepts of the physical layout have been discussed elsewhere.

A central task in diffraction analysis is the assignment of physical locations (3D vectors) to each detector pixel.  Actually, our task is two-fold:

1) Transform the data found on disk or in RAM to a useful numpy array.
2) Create a map from numpy array elements to physical locations.

The :class:`PADGeometry <bornagain.detector.PADGeometry>` contains the needed information to perform step (2).  This class also includes convenience methods that calculate commonly used quantities such as pixel solid angles, scattering vectors, and so on.  Once you have a :class:`PADGeometry <bornagain.detector.PADGeometry>` instance and a corresponding numpy array, it is (hopefully) much easier to perform your analysis tasks due to the standardized interface.  The main hurdle is often the layer of code that is needed to transplant raw data from a facility into the standard bornagain data containers.  That is what we will discuss below.

Before moving on, there is one caveat that needs to be mentioned now.  Since XFELs tend to use multiple PADs, you should plan to work with lists of :class:`PADGeometry <bornagain.detector.PADGeometry>` instances rather than a single one.  Most XFELs have segmented detectors that consist of several independent PADs.


Generic description of data layout
----------------------------------

Data stored on disk may be thought of as a finite 1D array, and our task is to chop it up into chunks of data corresponding to individual PADs.  We assume here that you are able to convert data on disk or RAM into a numpy array -- there are of course various details such as `data type <https://en.wikipedia.org/wiki/Data_type>`_ and `endianness <https://en.wikipedia.org/wiki/Endianness>`_ but usually there is a package such as `psnaa <https://confluence.slac.stanford.edu/display/PSDM/LCLS+Data+Analysis>`_ that does this work for you, or the data comes in a well-documented and well-supported format like `hdf5 <https://support.hdfgroup.org/HDF5/>`_ in which case we may use a package like `h5py <https://www.h5py.org/>`_ to load the data.  

In the most generic case, where we have a 1D data array that we wish to convert into individual 2D PAD data arrays, we need a few things:

1) The size of the 2D array we intend to extract, which we refer to as n_fs and n_ss for fast-scan and slow-scan directions.
1) The index of the first entry that corresponds to the corner pixel.
2) The fast-scan stride and the slow-scan stride.




Working with CrystFEL geometry files
------------------------------------

Firstly, you need to read about the CrystFEL `geom <http://www.desy.de/~twhite/crystfel/manual-crystfel_geometry.html>`_ 
file specification.  A few comments:

- CrystFEL geom files contain more than geometry information.  
- geom files contain information about the detector, regarding e.g. saturation levels, common-mode noise and conversions between digital data units and deposited x-ray energy.
- geom files contain information about the formatting of the files that contain the data
- geom files contain information that affects the behaviour of programs like indexamajig (e.g. the no_index card)

If you want the complete information in the geom file you can convert it to a python dictionary using the :func:`geometry_file_to_dict() <bornagain.external.cyrstfel.geometry_file_to_dict>` function, which is just a wrapper for the corresponding function in the cfelpyutils package.

Most importantly, geom files contain the three principal vectors that bornagain utilizes, albeit it may not be obvious at first glance when you look into the geom file.  If you just want this information, then you can simply use a geom file to generate a list :class:`PADGeometry <bornagain.detector.PADGeometry>` instances. 

If you have a CrystFEL geom file, you can load all the information into a python dictionary object with the :func:`bornagain.external.crystfel.geometry_file_to_dict()` function.

