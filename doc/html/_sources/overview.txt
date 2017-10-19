Overview
========

Here you will find an overview of basic concepts, which hopefully help you understand how the modules in bornagain are meant to fit together.

Some basics
-----------

- All units in bornagain are SI.  Angles are radians.  There are no known exceptions so far.
- All vectors should have a numpy shape of Nx3, in order to keep vector components close in memory.  We are quite strict about this.
- Detector panels are often segmented in real experiments.  The classes in the detector module are intended to help ease the burden of working with such detectors.
- Learn how to use iPython to explore the functionality of bornagain classes.
- If documentation is missing, tell someone (e.g. rkirian at asu dot edu).

Pixel-Array Detectors
---------------------

The most common detector we use is the "pixel-array detector", abbeviated as "PAD" in the bornagain package.  This detector is like a CCD and is assumed to consist of an orthogonal 2D grid of pixels.  This grid is described by the following vectors:

    :math:`\vec{T}` is the vector pointing from the origin to the center of the corner detector pixel that is assumed to be the first in memory.

    :math:`\vec{F}` is the vector that points along the "fast-scan" direction.  This is the distance and direction that points to the next pixel that is adjacent in physical space, as well as in computer memory.  The length of this vector indicates the pixel size.
    
    :math:`N_F` is the number of fast-scan pixels in the detector.
    
    :math:`\vec{S}` is the vector that points along the "slow-scan" direction.  This is much like the :math:`\vec{F}` vector, but these pixels are only adjacent in physical space but not in computer memory.  In computer memory, adjacent pixels have a memory stride of length :math:`N_F`.
    
    :math:`N_S` is the number of slow-scan pixels in the detector.

Note that there are no angles involved in describing the detector geometry.  That is because angles are confusing due to the many different conventions used by different reference books and software.  Also, importantly, rotation operations do not commute, which only adds to the confusion.

The incident x-ray beam direction is not assumed in bornagain.  We must specify this vector:

    :math:`\hat{B}` is the unit vector pointing along the incident beam direction.  This of course only makes sense if the illumination is a plane wave, but we take this to be the "nominal" beam direction.  In some places bornagain might default to the (0,0,1) direction, but you need not adhere to this.  Most likely it is only important for display purposes.
    
In the case of free-electron lasers, beam polarization is important:

    :math:`\hat{U}` is the polarization vector for the x-ray beam.  This single vector is appropriate for linearly polarized beams.  For beams that are not purely linearly polarized, one can sum the contributions from the two polarizations.  The polarization vector of the first component, along with the beam direction, specifies the second polarization vector.

With the above vectors specified, we may now generate the quantities that will be useful when doing diffraction analysis and simulations.  The vector pointing from the origin (where the target is located) to a detector pixel indexed by :math:`i` and :math:`j`, is 

    :math:`\vec{V}_{ij}=\vec{T}+i\vec{F}+j\vec{S}`

Easy, right?  Now let's compute the scattering vector for pixel :math:`i,j`:

    :math:`\vec{Q}_{ij}=\frac{2\pi}{\lambda}\left(\hat{V}_{ij} - \hat{B}\right)`

where :math:`\lambda` is the photon wavelength.  Next we can compute the scattering angle of a pixel:

    :math:`\theta_{ij} = \arccos(\hat{V}_{ij}\cdot\hat{B})`

For linearly polarized light, the polarization correction is

    :math:`P_{ij} = 1 - |\hat{U}\cdot\hat{V}_{ij}|^2`

If the light is not linearly polarized, then the polarization factor is a weighted sum of the above component and this one:

    :math:`P'_{ij} = 1 - |(\hat{B}\times\hat{U})\cdot\hat{V}_{ij}|^2`

The solid angle of a pixel is approximately equal to 

    :math:`\Delta \Omega_{ij} \approx \frac{\text{Area}}{R^2}\cos(\theta) = \frac{|\vec{F}\times\vec{S}|}{|V|^2}\hat{n}\cdot \hat{V}_{ij}`

where the vector normal to the PAD is 

    :math:`\hat{n} = \frac{\vec{F}\times\vec{S}}{|\vec{F}\times\vec{S}|}`

In the bornagain detector and source modules, you will find a reasonably good correspondence between the quantities above and the members of the classes within the :any:`detector` and :any:`source` modules.  

Simulation
----------

Some description of how we work with opencl.