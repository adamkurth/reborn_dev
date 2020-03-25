Beams
===========

The reborn :mod:`source <reborn.source>` package provides the :class:`Beam <reborn.source.Beam>` class for
describing x-ray beams.  This class may seem like overkill since the wavelength is often the only parameter that one
really needs to specify, but since reborn allows for an arbitrary incident beam direction you also need to know the
beam direction in addition to wavelength, at minimum.  The :class:`Beam <reborn.source.Beam>` class helps keep such
information together in a tidy (and standard) format.  If you are doing simulations, then you
probably also need to know the beam fluence (photons/area), and perhaps polarization.  In a more deluxe simulation,
you might also want to know the beam divergence and spectral width.  The :class:`Beam <reborn.source.Beam>` class
becomes increasingly helpful when you need to organize all this information, and it can save you a bit of typing when
converting between e.g. photons/area and energy/area.  The minimum parameters that are needed to describe a beam
are:

    :math:`E` : The nominal photon energy (the peak value of the beam has spectral width)

or, alternatively,

    :math:`\lambda` : The nominal photon wavelegnth (the peak value of the beam has spectral width)

Photon energy might be accompanied by a FWHM spread in photon energy :math:`\Delta E/E`, and the nominal beam direction
might be accompanied by the beam divergence FWHM.

Another important parameter is the beam direction:

    :math:`\hat{k}_0` : The incident beam direction.

Code in the reborn package does not in general assume a particular beam direction, but the
:class:`Beam <reborn.source.Beam>` class does set the default to :math:`\hat{k}_0 = (0, 0, 1)` which we might call
the :math:`\hat{z}` direction.   Note that virtually all concievable coordinate systems have been used by at least one
popular diffraction analysis package, which is why reborn focuses on vectorized math that removes all assumptions of
the beam direction.  There is, however, one place where a beam direction is presently assumed: the diffraction viewer
interfaces presently assume the :math:`\hat{k}_0 = (0, 0, 1)` beam direction (but this will be more flexible in the
future).

Beam polarization can also be important if you care about differential scattering cross sections.  The polarization axis
is defined as

   :math:`\hat{E}_0` : "Polarization axis" that defines a unique axis through which polarization is specified.

This single vector :math:`\hat{E}_0` is appropriate for linearly polarized beams, and the default value is
:math:`\hat{E}_0 = (1, 0, 0)`.  For beams that are not purely linearly polarized, you will most likely sum the intensity
contributions from each of the two polarizations, and the weights of those two contributions may be specified within the
:class:`Beam <reborn.source.Beam>` class.  The second polarization vector is not specified explicitly; it is *always*
equal to :math:`\hat{u}\times\hat{b}` due to the nature of Elecrodynamics. We do not presently support circularly or
elliptically polarized light but that can be added if need be.

Note that the :class:`Beam <reborn.source.Beam>`
class merely *specifies* information about an x-ray beam -- it is passed on to various functions that make use of that
information when doing calculations.  It is just a convenience.

You can find more information about what this class does by looking at the API documentation.
