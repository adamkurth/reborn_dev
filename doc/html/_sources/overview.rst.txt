Overview
========

Some day this page will have sufficient information to get you started with bornagain.

Some basics
-----------

- You must first learn the basics of `Python <https://www.python.org/>`_, preferably including the basic principles of object-oriented programming.
- It is recommended that you use Python 3.6 or later.
- The `tab-completion feature <https://ipython.org/ipython-doc/3/interactive/tutorial.html#tab-completion>`_ of `iPython <https://ipython.org/>`_ is one of the most efficient ways to explore the functionality of bornagain classes.
- The `numpy <http://www.numpy.org/#>`_ package is central to bornagain.
- All units in bornagain are SI.  Angles are radians.  There are no known exceptions so far.
- Everything is vectorized in bornagain.  Terms like "row", "column", "x", "y", or "z" have no official meaning.
- If documentation is missing or confusing, please fix it or tell someone (e.g. rkirian at asu dot edu).

X-ray beams
-----------

The bornagain package provides the :any:`source.Beam` class for describing x-ray beams.  So far it is a lightweight and minimalistic description of an x-ray beam.  The first couple of parameters that are needed to describe a beam are:

   :math:`\lambda` : the "nominal" wavelength of the beam

and

   :math:`\hat{b}` : the unit vector pointing along the "nominal" incident beam direction.

Wavelength might be accompanied by a FWHM spread in photon energy :math:`\Delta E/E`, and the nominal beam direction might be accompanied by the beam divergence FWHM.  The bornagain package does not make a general assumption about the beam direction, but the [0,0,1] direction is most commonly used so far.

Beam polarization can also be important:


   :math:`\hat{u}` is the polarization vector for the x-ray beam.

This single vector is appropriate for linearly polarized beams.  For beams that are not purely linearly polarized, one can sum the contributions from each of the two polarizations.  The second polarization vector is of course :math:`\hat{u}\times\hat{b}` .

Most of the above parameters can be specified by an instance of the :any:`source.Beam` class.  Derived quantities such as the polarization correction are tied to classes contained in the :any:`detector` module.


Pixel-Array Detectors
---------------------

.. figure:: figures/pad.jpg
   :scale: 80 %
   :alt: Pixel-array detector schematic

   Schematic of a Pixel-Array Detector.

The :any:`PADGeometry` class contains the data and methods needed to deal with "pixel-array detectors".  This detector is like a CCD and is assumed to consist of an orthogonal 2D grid of pixels.  The 2D grid is described by the following vectors:

   :math:`\vec{t}` is the vector pointing from the origin to the center of the corner detector pixel that is assumed to be the first in memory.

   :math:`\vec{f}` is the vector that points along the "fast-scan" direction.  This is the distance and direction that points to the next pixel.

that is adjacent in physical space, as well as in computer memory.  The length of this vector indicates the pixel size.
    
    :math:`n_f` is the number of fast-scan pixels in the detector.
    
    :math:`\vec{s}` is the vector that points along the "slow-scan" direction.  This is much like the :math:`\vec{f}` vector, but these pixels are only adjacent in physical space but not in computer memory.  In computer memory, adjacent pixels have a memory stride of length :math:`n_f`.
    
    :math:`n_s` is the number of slow-scan pixels in the detector.

Note that there are no angles involved in describing the detector geometry.  That is because angles are confusing due to the many different conventions used by different reference books and software.  Also, importantly, rotation operations do not commute, which only adds to the confusion.

With the above vectors specified, we may now generate the quantities that will be useful when doing diffraction analysis and simulations.  The vector pointing from the origin (where the target is located) to a detector pixel indexed by :math:`i` and :math:`j`, is 

    :math:`\vec{v}_{ij}=\vec{t}+i\vec{f}+j\vec{s}`

Now let's compute the scattering vector for pixel :math:`i,j`:

    :math:`\vec{q}_{ij}=\frac{2\pi}{\lambda}\left(\hat{v}_{ij} - \hat{b}\right)`

where :math:`\lambda` is the photon wavelength.  Next we can compute the scattering angle of a pixel:

    :math:`\theta_{ij} = \arccos(\hat{v}_{ij}\cdot\hat{b})`

For linearly polarized light, the polarization correction is

    :math:`P_{ij} = 1 - |\hat{u}\cdot\hat{v}_{ij}|^2`

If the light is not linearly polarized, then the polarization factor is a weighted sum of the above component and this one:

    :math:`P'_{ij} = 1 - |(\hat{b}\times\hat{u})\cdot\hat{v}_{ij}|^2`

The solid angle of a pixel is approximately equal to 

    :math:`\Delta \Omega_{ij} \approx \frac{\text{Area}}{R^2}\cos(\theta) = \frac{|\vec{f}\times\vec{s}|}{|v|^2}\hat{n}\cdot \hat{v}_{ij}`

where the vector normal to the PAD is 

    :math:`\hat{n} = \frac{\vec{f}\times\vec{s}}{|\vec{f}\times\vec{s}|}`

The :any:`PADGeometry` class can currently generate the above quantities for you.  More can be added if they are necessary.

Simulation
----------

The most basic starting point for a diffraction pattern from a collection of atoms is:

   :math:`I(\vec{q}) = I_0 r_e^2 |F(\vec{q})|^2 \Delta\Omega`

where the overall structure factor is

   :math:`F(\vec{q}) = \sum_n f_n(q)e^{i\vec{q}\cdot\vec{r}_n}`

and :math:`f_n(q)` is the wavelength-dependent scattering factor for atom :math:`n`.

More will follow from here.