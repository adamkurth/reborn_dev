Density Maps
============

.. _working_with_maps:

We often need to work with electron/scattering density maps or scattering intesity/amplitude maps in two or three
dimensions.
We will refer to all types of maps as "density maps" here.
In working with maps, there are different coordinate representations that we frequently
encounter, and there is a frequent need to extract 2D slices from 3D maps (for example by trilinear interpolation) or
to insert 2D slices into 3D maps (for example, by histogramming).
There is also the need to work with standard coordinates that arise due to algorithms such as the Fast Fourier
Transform.
Here we discuss the conventions that will be used consistently within reborn, and some of the tools that are
available for manipulating density maps.

Sampling and Binning
--------------------

.. _binning:

.. figure:: figures/bins_schematic.png
    :scale: 50 %
    :alt: 6 bin discrete
    :align: center

    Schematic of a 1D discretization with 6 bins.

In some cases a density maps corresponds to *points* at which a density map is *sampled*.
In other cases, density maps refer to *bins*, for example when data are merged or histogrammed.
In the case of binning, our standard convention for specifying coordinates is illustrated in the above figure.
For each dimension, we provide the total number of bins along the edge, :math:`N_{\mathrm{bin}}`, along with
the minimum and maximum values along each edge, :math:`x_{\mathrm{min}}`, :math:`x_{\mathrm{max}}`.
These coordinates refer to the *centers* of the bins, *not the edges*.

By the above definitions, the width of each bin is

.. math:: \Delta x = \frac{x_{\mathrm{max}} - x_{\mathrm{min}}}{N_{\mathrm{bin}} - 1}

The center position for the :math:`n` th bin is

.. math:: x_n = x_{\mathrm{min}} + n \Delta x

where the bin indices range from :math:`n = 0, \cdots, N_{\mathrm{bin}}-1`.  The bin index :math:`n_{\mathrm{s}}` for a
sample at position :math:`x_{\mathrm{s}}` can be calculated via

.. math:: n_{\mathrm{s}} = \mathrm{floor} \left( \frac{x_{\mathrm{s}} - (x_{\mathrm{min}} - \Delta x / 2)}{\Delta x} \right)

Note that the above means that (1) If :math:`x_{\mathrm{s}} = x_{\mathrm{min}} - \Delta x /2` the index should be zero.
(2) If :math:`x_{\mathrm{s}} = x_{\mathrm{min}} + \Delta x /2` the index should be one.  This is an arbitrary choice,
but it is important that we be consistent.

Data layout for multi-dimensional maps
--------------------------------------

We base our indexing of density maps on numpy conventions.  By default, numpy uses "c-contiguous order", which
means that incremements in the right-most index of an array correspond to the smallest steps in the internal memory
buffer.
In reborn, it is assumed that all arrays are c-contiguous as with the case of arrays of vectors
:ref:`decribed elsewhere <arrays_of_vectors>`.
The "c-contiguous" distinction does not matter in some calculations, but it certainly does if you pass an array to a
reborn function that is compiled from underlying OpenCL code or Fortran code.

A particular sample in a 3D density map has a 3-tuple of indices :math:`(i, j, k)`, and the array size is
expressed by a 3-tuple :math:`(N_{\mathrm{bin}}^i,N_{\mathrm{bin}}^j,N_{\mathrm{bin}}^k)` that corresponds to the numpy
"shape" attribute.
Similarly, we have corresponding 3-tuples of :math:`x_{\mathrm{min}}` and :math:`x_{\mathrm{max}}`.
We often need to create an :math:`M\times 3` array of position vectors that correspond to the density samples (where
:math:`M = N_{\mathrm{bin}}^iN_{\mathrm{bin}}^jN_{\mathrm{bin}}^k` is the length of the "flattened" 3D numpy array).
The example below shows one way to do this:

.. testcode::

    import numpy as np
    shape = np.array([2, 3, 4])
    x_min = np.array([-5, -2.5,  0])
    x_max = np.array([+5, +2.5, 15])
    dx = (x_max-x_min)/(shape-1)
    x = np.arange(shape[0])*dx[0]+x_min[0]
    y = np.arange(shape[1])*dx[1]+x_min[1]
    z = np.arange(shape[2])*dx[2]+x_min[2]
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    # Note the optional indexing='ij', because the meshgrid default gives bizarre output
    r = np.vstack([xx.ravel(),yy.ravel(),zz.ravel()]).T.copy()
    # Note: the .copy() part in the above line makes a c-contiguous array
    assert np.all(xx.shape == np.array(shape))
    print(r.flags.c_contiguous)
    print(r[0:12,:].T)

.. testoutput::

    True
    [[-5.  -5.  -5.  -5.  -5.  -5.  -5.  -5.  -5.  -5.  -5.  -5. ]
     [-2.5 -2.5 -2.5 -2.5  0.   0.   0.   0.   2.5  2.5  2.5  2.5]
     [ 0.   5.  10.  15.   0.   5.  10.  15.   0.   5.  10.  15. ]]

In the final output of the above example, one might say that the ":math:`z`" vector component (i.e. third component)
increments fastest in memory, which might go against your intuition if you assumed that ":math:`x` should ingrement
faster than :math:`z`" for various reasons.
It is also noteworthy that the numpy
`meshgrid function <https://docs.scipy.org/doc/numpy/reference/generated/numpy.meshgrid.html>`_
allows for both so-called "matrix indexing" and "Cartesian indexing", and the default behavior of meshgrid is to
scramble the the ordering of indices without a trace of reasoning in the documentation.
Importantly, there is no notion of ":math:`x`", ":math:`y`" and ":math:`z`" coordinates in reborn -- the only
important matter here is that we be consistent in the way that we order vector components, and the above example
defines the ordering that is assumed everywhere in reborn.  This ordering derives from the numpy indexing
conventions (not the bizarre default behavior of meshgrid).

Saving density maps
-------------------

.. _nd_array_handling:

**numpy format**:  If we choose to save in numpy compressed format with ".npz" extension, we agree to the following
rules.
There are at least three types of densities that we routinely deal with in reborn (see e.g.
:ref:`working with crystals <working_with_crystals>` ): electron/scattering density (possibly complex), diffraction
amplitude (usually complex), and diffraction intensity (always real).
We specify the type of density by the variable "type", which is a string that is equal to "density", "amplitude", or
"intensity".  There are four coordinate bases that we routinely use, which correspond to cartesian real space
coordinates :math:`\vec{r}`, crystallographic fractional coordinates :math:`\vec{x}`, reciprocal space coordinates
:math:`\vec{q}`,
or Miller indices :math:`\vec{h}`.  Within the npz file, we specify the basis by including a variable named
"representation" that may be equal to one of the four strings "r", "x", "q", or "h".
We then include "map_min" and "map_max" to specify :math:`x_{\mathrm{min}}` and :math:`x_{\mathrm{max}}` as defined above.
The actual map should be saved as the variable named "map", and its shape corresponds to :math:`N_{\mathrm{bin}}`.


Slicing and inserting
---------------------

This section will follow -- should explain how we go about extracting 2D slices by e.g. trilinear interpolation, and
also how we insert slices as when we merge 2D diffraction intensities into 3D maps.