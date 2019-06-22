Detectors
=========

Pixel-Array Detectors
---------------------

.. figure:: figures/pad.jpg
    :scale: 80 %
    :alt: Pixel-array detector schematic

    Schematic of a Pixel-Array Detector.

The :class:`PADGeometry <bornagain.detector.PADGeometry>` class contains the data and methods needed to deal
with "pixel-array detectors" (PADs).  This
detector is like a CCD and is assumed to consist of an orthogonal 2D grid of pixels.  The 2D grid is described by the
following vectors:

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

The :any:`PADGeometry` class can currently generate the above quantities for you, along with other helpful functions.


Data and geometry formats
-------------------------

A central task in diffraction analysis is the assignment of physical locations (3D vectors) to each detector pixel.
Actually, our task is two-fold:

1) Transform the data found on disk or in RAM to a useful numpy arrays.
2) Determine the vectors that specify 3D positions corresonding to the elements of the numpy arrays.

The :class:`PADGeometry <bornagain.detector.PADGeometry>` class contains the needed information to perform step (2), but
does not have any involvement in step (1).  Step (1) is often a messy process that requires specialized code, and
therefore we make no effort to standardize that process.  However, once you have a
:class:`PADGeometry <bornagain.detector.PADGeometry>` instance along with corresponding numpy arrays, your analsis code
can hopefully be written in a source-agnostic way.

Since XFELs tend to use multiple PADs, you should plan to work with lists of
:class:`PADGeometry <bornagain.detector.PADGeometry>` instances rather than a single one. You can still do vectorized
operations on all panels at once with the numpy ravel function.


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
:func:`load_crystfel_geometry() <bornagain.external.crystfel.load_crystfel_geometry>` function, which is just a wrapper
for the corresponding function in the `cfelpyutils <https://pypi.org/project/cfelpyutils/>`_ package.

Most importantly, geom files contain the three principal vectors that bornagain utilizes, albeit it may not be obvious
at first glance when you look into the geom file.  If you just want this information, then you can simply use a geom
file to generate a list of :class:`PADGeometry <bornagain.detector.PADGeometry>` instances via the
:func:`geometry_file_to_pad_geometry_list() <bornagain.external.crystfel.geometry_file_to_pad_geometry_list>` function.