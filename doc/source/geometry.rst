.. _doc_detectors:

Detectors
=========

.. _doc_pads:

Pixel-Array Detectors
---------------------

.. figure:: figures/detector_geometry.svg
    :scale: 80 %
    :alt: Pixel-array detector schematic

    Schematic of a Pixel-Array Detector.

The |PADGeometry| class contains the data and methods needed to deal with "pixel-array detectors" (PADs).
This detector is assumed to consist of a regular 2D grid of pixels specified by the following parameters:

* :math:`\vec{t}` is the vector that points from the origin to the *center* of the first pixel in memory.
* :math:`\vec{f}` is the vector that points from the first pixel in memory to the next pixel in the fast-scan direction.
* :math:`\vec{s}` The vector that points from the first pixel in memory, to the next pixel in the slow-scan direction.
* :math:`n_f` is the number of pixels along the fast-scan direction.
* :math:`n_s` is the number of pixels along the slow-scan direction.

In the above:

* The :math:`\vec{f}` and :math:`\vec{s}` vectors form the basis of the 2D grid of pixels.  As such, they also
  define the "pixel size".
* The term "fast-scan" corresponds to the right-most index of an |ndarray| containing PAD data.
* The term "slow-scan" corresponds to the left-most index of an |ndarray| containing PAD data.
* In the default memory buffer layout of an |ndarray|, the fast-scan direction corresponds to pixels that are
  contiguous in memory, and which have the smallest stride.  If the phrase "contiguous in memory" and the
  term "stride" does not mean anything to you, then you need to read the |numpy| documentation for |ndarray|.

.. note::

    The reborn package never uses angles to describe detector geometry.  Angles cause a lot of confusion that is easily
    avoided with the use of vectors.

Additional vectors that are important for calculating things related to x-ray scattering, but which are not inherently
related to detector geometry, are:

:math:`\hat{b}` is the incident beam vector that describes the nominal direction of the x-ray beam.

:math:`\hat{e}_1` is the direction of the principle electric field vector, which would be the direction of the electric
field in the case of linearly polarized x-rays.  The second vector that is needed for unpolarized or elliptically
polarized x-rays is always :math:`\hat{e}_2 = \hat{b}\times\hat{e}_1` .

With the above vectors specified, we may now generate the quantities that will be useful when doing diffraction analysis
and simulations.  Central to many calculations is the vector pointing from the origin (where the target is located) to a
detector pixel indexed by :math:`i` and :math:`j`, is

.. math::

    \vec{v}_{ij}=\vec{t}+i\vec{f}+j\vec{s}

Note that the above associates the :math:`i` index with the fast scan.  Therefore, the correct way to access this
element in a numpy array is `data[j, i]`, because the right-most index is the fast-scan index (for an |ndarray| that
is in the default "c-contiguous" layout).

Now let's compute the scattering vector for pixel :math:`i,j`:

.. math::

    \vec{q}_{ij}=\frac{2\pi}{\lambda}\left(\hat{v}_{ij} - \hat{b}\right)

where :math:`\lambda` is the photon wavelength.  Next we can compute the scattering angle of a pixel:

.. math::

    \theta_{ij} = \arccos(\hat{v}_{ij}\cdot\hat{b})

For linearly polarized light, the polarization correction is

.. math::

    P_{ij} = 1 - |\hat{e}_1\cdot\hat{v}_{ij}|^2

If the light is not linearly polarized, then the polarization factor is a weighted sum of the above component and this
one:

.. math::

    P'_{ij} = 1 - |(\hat{b}\times\hat{e}_1)\cdot\hat{v}_{ij}|^2

The solid angle of a pixel is approximately equal to

.. math::

    \Delta \Omega_{ij} \approx \frac{\text{Area}}{R^2}\cos(\theta) = \frac{|\vec{f}\times\vec{s}|}{|v|^2}\hat{n}\cdot \hat{v}_{ij}`

where the vector normal to the PAD is

.. math::

    \hat{n} = \frac{\vec{f}\times\vec{s}}{|\vec{f}\times\vec{s}|}

The |PADGeometry| class can currently generate the above quantities for you, along with other helpful quantities.  The
|PADGeometryList| class combines multiple |PADGeometry| instances.

.. note::

    Once the above is understood, you might want to look at the :ref:`example <example_pad_geometry>` of how to use the
    PAD geometry tools provided by reborn.

Working with multiple PADs
--------------------------

XFELs frequently use detectors that are split up in to many separate PADs.  The |PADGeometryList| class is a special
sub-class of the python list that provides convenience methods for working with multiple PADs.  Some
:ref:`examples <example_pad_geometry>` are provided.

Data and slicing
----------------

By default, |PADGeometry| and |PADGeometryList| assume that the data for each PAD is stored in contiguous memory blocks.
However, there are many cases in which PAD data are not contiguous due to hardware considerations.
For example, 4 PADs on a single silicon chip might be stored in a 2x2 arrangement in order to maximize read/write
speeds, but each PAD is not contiguous.

If your raw data is not contiguous, then you probably have an important decision to make:

**Option 1** is to write a specialized functions that extract all the PAD data arrays and make them contiguous before
passing them into your analysis pipeline.
This is good if you care about having an analysis pipeline that is agnostic to the origin of the data, and which can
easily handle mixtures of differently sized PADs.

**Option 2** is to maintain the raw data layout throughout your analysis.  This is good if you want to easily save
processed data in the same layout as the raw data, and if you have some geometry files that refer to the raw data
layout.

If you wish to maintain the raw data layout, you may configure a |PADGeometry| instance to contain information about how
the individual PAD data should be |sliced| from the parent |ndarray|.
This is specified by the following parameters:

    * ``parent_data_shape`` : The expected shape of the parent |ndarray|.
    * ``parent_data_slice`` : The |slice| object needed to extract or insert this PAD's data into the parent |ndarray|.

There are |PADGeometry| methods such as ``slice_from_parent`` that might make your code a bit cleaner.

Reading and writing PAD geometry info
-------------------------------------

There are methods in the detector class for reading and writing the information needed to save/recall a
|PADGeometryList|.  They are currently saved in json format, but this will likely change now that we know that json
files do not accommodate comments.

Working with CrystFEL geometry files
------------------------------------

reborn includes a module to help with reading CrystFEL
`geom <http://www.desy.de/~twhite/crystfel/manual-crystfel_geometry.html>`_ files.  If you just want a |PADGeometryList|
then you can simply use the
:func:`geometry_file_to_pad_geometry_list() <reborn.external.crystfel.geometry_file_to_pad_geometry_list>` function.
Note that the ``parent_data_shape`` and ``parent_data_slice`` attributes will be set by this function.

CrystFEL geom files contain a lot more than static geometry information.  They also contain information about

* detector properties (e.g. saturation levels, common-mode noise and conversions between digital data units and
  deposited x-ray energy),
* information about how to obtain encoder values that specify detector positions,
* formatting of the files that contain the diffraction data, and
* how programs like indexamajig should treat the data (e.g. the no_index card)

If you want to read in the complete information from a geom file you can convert it to a python dictionary using the
:func:`load_crystfel_geometry() <reborn.external.crystfel.load_crystfel_geometry>` function, which is just a wrapper
for the corresponding function in the `cfelpyutils <https://pypi.org/project/cfelpyutils/>`_ package.
