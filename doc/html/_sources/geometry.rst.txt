Detectors
=========

Pixel-Array Detectors
---------------------

.. figure:: figures/pad.jpg
    :scale: 80 %
    :alt: Pixel-array detector schematic

    Schematic of a Pixel-Array Detector.

The :any:`PADGeometry` class contains the data and methods needed to deal with "pixel-array detectors".  This detector
is like a CCD and is assumed to consist of an orthogonal 2D grid of pixels.  The 2D grid is described by the following
vectors:

    :math:`\vec{t}` is the vector pointing from the origin to the center of the corner detector pixel that is assumed to
    be the first in memory.

    :math:`\vec{f}` is the vector that points along the "fast-scan" direction.  This is the distance and direction that
    points to the next pixel. that is adjacent in physical space, as well as in computer memory.  The length of this
    vector indicates the pixel size.

    :math:`n_f` is the number of fast-scan pixels in the detector.

    :math:`\vec{s}` is the vector that points along the "slow-scan" direction.  This is much like the :math:`\vec{f}`
    vector, but these pixels are only adjacent in physical space but not in computer memory.  In computer memory,
    adjacent pixels have a memory stride of length :math:`n_f`.

    :math:`n_s` is the number of slow-scan pixels in the detector.

Note that there are no angles involved in describing the detector geometry.  That is because angles are confusing due
to the many different conventions used by different reference books and software.  Also, importantly, rotation
operations do not commute, which only adds to the confusion.

With the above vectors specified, we may now generate the quantities that will be useful when doing diffraction analysis
and simulations.  The vector pointing from the origin (where the target is located) to a detector pixel indexed by
:math:`i` and :math:`j`, is

    :math:`\vec{v}_{ij}=\vec{t}+i\vec{f}+j\vec{s}`

Now let's compute the scattering vector for pixel :math:`i,j`:

    :math:`\vec{q}_{ij}=\frac{2\pi}{\lambda}\left(\hat{v}_{ij} - \hat{b}\right)`

where :math:`\lambda` is the photon wavelength.  Next we can compute the scattering angle of a pixel:

    :math:`\theta_{ij} = \arccos(\hat{v}_{ij}\cdot\hat{b})`

For linearly polarized light, the polarization correction is

    :math:`P_{ij} = 1 - |\hat{u}\cdot\hat{v}_{ij}|^2`

If the light is not linearly polarized, then the polarization factor is a weighted sum of the above component and this
one:

    :math:`P'_{ij} = 1 - |(\hat{b}\times\hat{u})\cdot\hat{v}_{ij}|^2`

The solid angle of a pixel is approximately equal to

    :math:`\Delta \Omega_{ij} \approx \frac{\text{Area}}{R^2}\cos(\theta) = \frac{|\vec{f}\times\vec{s}|}{|v|^2}\hat{n}\cdot \hat{v}_{ij}`

where the vector normal to the PAD is

    :math:`\hat{n} = \frac{\vec{f}\times\vec{s}}{|\vec{f}\times\vec{s}|}`

The :any:`PADGeometry` class can currently generate the above quantities for you.  More can be added if they are
necessary.


Data and geometry formats
-------------------------

A central task in diffraction analysis is the assignment of physical locations (3D vectors) to each detector pixel.
Actually, our task is two-fold:

1) Transform the data found on disk or in RAM to a useful numpy arrays.
2) Create a mapping from each pair numpy array indices to the corresponding position 3-vector.

The :class:`PADGeometry <bornagain.detector.PADGeometry>` contains the needed information to perform step (2).  This
class also includes convenience methods that calculate commonly used quantities such as pixel solid angles, scattering
vectors, and so on.  Once you have a :class:`PADGeometry <bornagain.detector.PADGeometry>` instance and a corresponding
numpy array, the standardized interface will make it easier to re-use code.

The main hurdle usually lies in step (1), for which we need layer of code to transplant raw data from a facility into
standard numpy arrays.  This is discussed in the next section.

Before moving on, there is one caveat that needs to be mentioned.  Since XFELs tend to use multiple PADs, you should
plan to work with lists of :class:`PADGeometry <bornagain.detector.PADGeometry>` instances rather than a single one.
You can still do vectorized operations on all panels at once with the numpy ravel function, as shown in some example
somewhere (TODO:rickk).


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

1) The size of the 2D array we intend to extract, which we refer to as :math:`n_{fs}` and :math:`n_{ss}` for fast-scan
   and slow-scan directions.
2) The index :math:`a` of the first datapoint in memory, assumed to correspond to a corner pixel.
3) The fast-scan stride, :math:`S_f`, and the slow-scan stride :math:`S_s`.

From the above, we can get the intensity value of pixel :math:`i,j` from the raw data array as follows:

:math:`PAD[i,j] = RAW[a + i*S_f + j*S_s]`

We've not had to deal with this general case yet, but when we do we can probably just use the numpy methods of dealing
with arbitrary strides.


Working with CrystFEL geometry files
------------------------------------

Firstly, you need to read about the CrystFEL `geom <http://www.desy.de/~twhite/crystfel/manual-crystfel_geometry.html>`_ 
file specification.  Note that CrystFEL geom files contain a lot more than geometry information.  They also contain
information about...

- detector properties (e.g. saturation levels, common-mode noise and conversions between digital data units and
  deposited x-ray energy),
- information about how to obtain encoder values that specify detector positions,
- formatting of the files that contain the diffraction data,
- how programs like indexamajig should treat the data (e.g. the no_index card)

If you want to read in the complete information from a geom file you can convert it to a python dictionary using the
:func:`geometry_file_to_dict() <bornagain.external.cyrstfel.geometry_file_to_dict>` function, which is just a wrapper
for the corresponding function in the `cfelpyutils <https://pypi.org/project/cfelpyutils/>`_ package.

Most importantly, geom files contain the three principal vectors that bornagain utilizes, albeit it may not be obvious
at first glance when you look into the geom file.  If you just want this information, then you can simply use a geom
file to generate a list of :class:`PADGeometry <bornagain.detector.PADGeometry>` instances for use in bornagain.