Beams
===========

The bornagain :mod:`source <bornagain.source>` package provides the :class:`Beam <bornagain.source.Beam>` class for
describing x-ray beams.  In some sense this class is overkill since we often need only know the wavelength, but since
bornagain does not specify a standard beam direction you also need to know the beam direction at minimum, and this
class helps keep such information together in a tidy and standard format.  If you are doing simulations, then you
probably also need to know the beam fluence (photons/area), and perhaps polarization.  In a more deluxe simulation,
you might also want to know the beam divergence and spectral width.  The :class:`Beam <bornagain.source.Beam>` class
exists to help organize all this information.  The first couple of
parameters that are needed to describe a beam are:

   :math:`\lambda` : the "nominal" wavelength of the beam

and

   :math:`\hat{b}` : the unit vector pointing along the "nominal" incident beam direction.

Wavelength might be accompanied by a FWHM spread in photon energy :math:`\Delta E/E`, and the nominal beam direction
might be accompanied by the beam divergence FWHM.  The bornagain package does not make a general assumption about the
beam direction, but the [0,0,1] direction is most commonly used (in particular, the viewers in bornagain assume that
to be the direction thus far).

Beam polarization can also be important:

   :math:`\hat{u}` is the polarization vector for the x-ray beam.

This single vector is appropriate for linearly polarized beams.  For beams that are not purely linearly polarized, one
can sum the contributions from each of the two polarizations.  The second polarization vector is equal to
:math:`\hat{u}\times\hat{b}` .

You can find more information about what this class does by looking at the API documentation.
