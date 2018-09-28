Beams
===========

The bornagain :mod:`source <bornagain.source>` package provides the :class:`Beam <bornagain.source.Beam>` class for
describing x-ray beams.  So far it is a lightweight and minimalistic description of an x-ray beam.  The first couple of
parameters that are needed to describe a beam are:

   :math:`\lambda` : the "nominal" wavelength of the beam

and

   :math:`\hat{b}` : the unit vector pointing along the "nominal" incident beam direction.

Wavelength might be accompanied by a FWHM spread in photon energy :math:`\Delta E/E`, and the nominal beam direction
might be accompanied by the beam divergence FWHM.  The bornagain package does not make a general assumption about the
beam direction, but the [0,0,1] direction is most commonly used so far.

Beam polarization can also be important:

   :math:`\hat{u}` is the polarization vector for the x-ray beam.

This single vector is appropriate for linearly polarized beams.  For beams that are not purely linearly polarized, one
can sum the contributions from each of the two polarizations.  The second polarization vector is of course
:math:`\hat{u}\times\hat{b}` .

So far we've not really used the Beam class... but eventually we will...

TODO: Add spectrum handling to beam class

