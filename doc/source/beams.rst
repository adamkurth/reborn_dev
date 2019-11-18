Beams
===========

The bornagain :mod:`source <bornagain.source>` package provides the :class:`Beam <bornagain.source.Beam>` class for
describing x-ray beams.  This class may seem like overkill since the wavelength is often the only parameter that one
really needs to specify, but since bornagain allows for an arbitrary incident beam direction you also need to know the
beam direction in addition to wavelength, at minimum.  The :class:`Beam <bornagain.source.Beam>` class helps keep such
information together in a tidy (and standard) format.  If you are doing simulations, then you
probably also need to know the beam fluence (photons/area), and perhaps polarization.  In a more deluxe simulation,
you might also want to know the beam divergence and spectral width.  The :class:`Beam <bornagain.source.Beam>` class
becomes increasingly helpful when you need to organize all this information, and it can save you a bit of typing when
converting between e.g. photons/area and energy/area.  The minimum parameters that are needed to describe a beam
are:

    :math:`E` : The nominal photon energy (the peak value of the beam has spectral width)

or, alternatively,

    :math:`\lambda` : The nominal photon wavelegnth (the peak value of the beam has spectral width)

Photon energy might be accompanied by a FWHM spread in photon energy :math:`\Delta E/E`, and the nominal beam direction
might be accompanied by the beam divergence FWHM.

Another important parameter is the beam direction:

    :math:`\hat{b}` : The incident beam direction.

Code in the bornagain package does not in general assume a particular beam direction, but the
:class:`Beam <bornagain.source.Beam>` class does set the default to [0, 0, 1].   If you think the "z" direction is the
only "obvious" choice for the beam direction, you should read the documentation for popular software such as MOSFLM.
There is one place where a beam direction is presently assumed in bornagain: the viewers presently assume the [0, 0, 1]
beam direction.

Beam polarization can also be important:

   :math:`\hat{u}` : Defines a unique axis through which polarization is specified.

This single vector :math:`\hat{u}` is appropriate for linearly polarized beams, and the default value is the [1, 0, 0]
direction.  For beams that are not purely linearly polarized, you will most likely sum the intensity contributions from
each of the two polarizations, and the weights of those two contributions may be specified within the
:class:`Beam <bornagain.source.Beam>` class.  The second polarization vector is not specified explicitly; it is *always*
equal to :math:`\hat{u}\times\hat{b}` due to the nature of Elecrodynamics. We do not presently support circularly or
elliptically polarized light but that can be added if need be.

Note that the :class:`Beam <bornagain.source.Beam>`
class merely *specifies* information about an x-ray beam -- it is passed on to various functions that make use of that
information when doing calculations.  It is just a convenience.

You can find more information about what this class does by looking at the API documentation.
