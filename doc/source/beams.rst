Beams
=====

The |Beam| class is the standard way to describing an x-ray beam.  The following parameters are presently used by
various functions in reborn:

    :math:`E` : The nominal photon energy

    :math:`\lambda` : The nominal photon wavelegnth

    :math:`\Delta E/E` : The nominal FWHM of photon energy

    :math:`\Delta \theta_b` : The nominal beam divergence FWHM

    :math:`\hat{k}_0` : The incident beam direction

    :math:`\hat{E}_1` : The primary polarization axis of the electric field

    :math:`\hat{E}_2` : The secondary polarization axis, equal to :math:`\hat{k}_0 \times \hat{E}_1`)

Code in the reborn package does not assume a "standard" beam direction, but the |Beam| class does set the
default to :math:`\hat{k}_0 = (0, 0, 1)` which we might call the :math:`\hat{z}` direction.   There is, however, one
place where a beam direction is presently assumed: the diffraction viewers presently assume the default beam direction.
