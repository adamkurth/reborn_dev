..

    Generic description of data layout
    ----------------------------------

    It's not likely that bornagain will have some generalized class for reading in data in arbitrary formats.  We'll
    eventually support the `cxidb <http://www.cxidb.org/>`_ file format, and for any other format someone must write a
    custom function.  But it's nonetheless worthwile to think a little bit about the task at hand.

    Data stored on disk may be thought of as a finite 1D array, and our task is to chop it up into chunks of data
    corresponding to individual pixel-array detectors.  The first step is probably to convert data on disk or RAM into a
    numpy array, and usually there is a package such as
    `psana <https://confluence.slac.stanfor.edu/display/PSDM/LCLS+Data+Analysis>`_ that does this work for you, or the data
    comes in a well-documented and well-supported format like `hdf5 <https://support.hdfgroup.org/HDF5/>`_ in which case
    we may use a package like `h5py <https://www.h5py.org/>`_.

    In the best of situations, we get a numpy array with some reasonable shape, and it's then easy to split up the block of
    contiguous data into a list of individual panels.  You don't need to copy memory; you can instead use numpy
    `views <https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.ndarray.view.html>`_ of the initial array.

    In the most generic case, where we have a 1D data array that we wish to convert into individual 2D PAD data arrays, we
    need a few things:

    1) The shape of the 2D array we intend to extract, which we refer to as :math:`n_{fs}` and :math:`n_{ss}` for fast-scan
       and slow-scan directions.
    2) The index :math:`a` of the first datapoint in memory, assumed to correspond to a corner pixel.
    3) The fast-scan stride, :math:`S_f`, and the slow-scan stride :math:`S_s`.

    From the above, we can get the intensity value of pixel :math:`i,j` from the raw data array as follows:

    :math:`PAD[i,j] = RAW[a + i*S_f + j*S_s]`

    We've not had to deal with this general case yet, but when we do we can probably just use the numpy methods of dealing
    with arbitrary strides.