Overview
========

Here you will find an overview of basic concepts, which hopefully help you understand how the modules in bornagain are meant to fit together.

Some basics
-----------

- All units in bornagain are SI.  Angles are radians.  Exceptions are very rare.
- All vectors should have a numpy shape of Nx3, in order to keep vector components close in memory.  We are quite strict about this.
- Detector panels are often segmented in real experiments.  The classes in the detector module are intended to help ease the burden of working with such detectors.
- Learn how to use iPython to explore the functionality of bornagain classes.
- If documentation is missing, tell someone (e.g. rkirian at asu dot edu).

Detectors
---------

There are a few detectors that we might want to work with.  So far we only deal with polar and rectangular type detectors.

The most commond detector is known as the "pixel-array detector", or "PAD", in the bornagain package.  This type of detector is assumed to consist of a 2D grid of pixels.  This grid is described by the following vectors:

    :math:`\vec{T}` is the vector pointing from the origin to the corner detector pixel that is first in memory.

    :math:`\vec{F}` is the vector that points along the "fast-scan" direction.  This is the distance and direction that points to the next pixel that is adjacent in physical space as well as in computer memory.  The length of this vector indicates the pixel size.
    
    :math:`N_F` is the number of fast-scan pixels in the detector.
    
    :math:`\vec{S}` is the vector that points along the "slow-scan" direction.  This is much like the :math:`\vec{F}` vector, but these pixels are only adjacent in physical space but not in computer memory.  In computer memory, adjacent pixels have a memory stride of length :math:`N_F`.
    
    :math:`N_S` is the number of slow-scan pixels in the detector.

Note that there are no angles involved in describing the physical position and orientation of a PAD.  That is because angles are confusing due to the many different conventions used by different books and software.  Since angles are not used in bornagain, we must specify the x-ray beam direction:

    :math:`\hat{B}` is the unit vector pointing along the incident beam direction.  This of course only makes sense if the illumination is a plane wave, but sometimes we might take this to be the "nominal" beam direction.  The convention in bornagain is to take this vector along the (0,0,1) direction, but you need not adhere to this.  Most likely it is only important for display purposes.
    
We must know a bit more about the incident illumination:

    :math:`\hat{U}` is the polarization vector for the x-ray beam.  This is appropriate for linearly polarized beams, and that is all that is handled by bornagain thus far.

With the above vectors specified, we may now generate the quantities that will be useful when doing diffraction analysis and simulations.  The vector pointing from the origin (where the target is located) to a detector pixel indexed by :math:`i` and :math:`j`, is 
    :math:`\vec{V}_{ij}=\vec{T}+i\vec{F}+j\vec{S}`
Easy, right?  Now let's compute scattering vector for pixel :math:`i,j`:
    :math:`\vec{Q}_{ij}=\frac{2\pi}{\lambda}\left(\hat{V}_{ij} - \hat{B}\right)`
where :math:`\lambda` is the photon wavelength.  Next we can comput the scattering angle of a pixel:
    :math:`\theta_{ij} = \arccos(\hat{V}_{ij}\cdot\hat{B})`
For linearly polarized light, the polarization correction is
    :math:`P_{ij} = 1 - |\hat{U}\cdot\hat{V}_{ij}|^2`
The solid angle of a pixel is approximately equal to 
    :math:`\Delta \Omega_{ij} \approx \frac{|\vec{F}\times\vec{S}|}{|V|^2}\hat{n}\cdot \hat{V}_{ij}`
where the vector normal to the PAD is 
    :math:`\hat{n} = \frac{\vec{F}\times\vec{S}}{|\vec{F}\times\vec{S}|}`

In the bornagain detector and source modules, you will find a reasonably good correspondence between the quantities above and the members of the classes within the :any:`detector` and :any:`source` modules.  

Simulation
----------

Some description of how we work with opencl.