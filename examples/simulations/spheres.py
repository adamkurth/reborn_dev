import sys
sys.path.append('../..')
from time import time

import numpy as np
import matplotlib.pyplot as plt

import bornagain
from bornagain import simulate
from bornagain.simulate.form_factors import sphere_form_factor

use_numba = False
try:
    # This prepares the GPU simulation engine.  It computes this sum: Sum_n f_n exp(i q.r_n)
    # (really fast -- overkill...)
    from bornagain.simulate.clcore import ClCore
    sim = simulate.clcore.ClCore()
    phase_factor_qrf = sim.phase_factor_qrf
except ImportError:
    use_numba = True

# use_numba = True
if use_numba:
    # Default to numba if pyopencl is not available
    from bornagain.simulate import numbacore
    phase_factor_qrf = numbacore.phase_factor_qrf


# We simulate diffraction from an ensemble of spheres.  Specifically, we have doublets of spheres sitting in a plane
# tilted 45 deg. with respect to the x-ray beam.  The real-space density is:
#    f_tot(r) = Sum_n f_sph(r - s_n)
# where s_n is the position of the n_th sphere.  The Fourier-space density is:
#    F_tot(q) = Sum_n F_sph(q) exp(i q.s_n)
# Notably, the form factor of the sphere, F_sph(q), can come out of the sum:
#    F_tot(q) = F_sph(q) [ Sum_n exp(i q.s_n) ]
# We simulate the form factor of the sphere once, and do the sum over the position phasors separately.
#
# Everything in SI
#
# We use a common convention at x-ray facilities:
#     - Beam is along the "z" axis (i.e. third vector compoment)
#     - The "y" component is "up"
#     - The "x" compoment completes the right-handed coordinate system
#     - Usually beam polarization from an undulator is in the horizontal, which would be "x"



# Generic container for an x-ray beam.  Holds beam direction, polarization, wavelength, etc.
beam = bornagain.source.Beam(wavelength=13.49e-9, beam_vec=[0, 0, 1])

# Generic container for an pixel-array detector
detector = bornagain.detector.PADGeometry(n_pixels=2048, pixel_size=13.5e-6, distance=0.07)

# Now we construct the arrangement of spheres:
radius = 0.5e-6
pos = np.array([[-5, -5, 0], [-5, 4, 0], [2, 0, 0]])*1e-6
pos = np.concatenate([pos, pos], axis=0)
sep = radius*1.6 # guess the sphere separation
pos[0:3, 2] = -sep
pos[3:6, 2] = +sep
angle = np.pi/4
rot = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
pos = np.dot(pos, rot.T)

# Display the arrangement of spheres:
if False:
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    p = pos*1e6
    ax.scatter(p[:, 0], p[:, 1], p[:, 2], c='k', marker='o') # matplotlib can't get basic 3d axis scaling right...
    ax.set_xlabel('x (microns)')
    ax.set_ylabel('y (microns)')
    ax.set_zlabel('z (microns)')
    plt.show()

# Compute the scattering amplitudes:
t = time()
scat = phase_factor_qrf(q=detector.q_vecs(beam=beam), r=pos, f=np.ones([pos.shape[0]]))
print(time() - t, ' seconds')
scat *= simulate.form_factors.sphere_form_factor(radius=radius, q_mags=detector.q_mags(beam=beam))
scat = np.abs(scat)**2
scat /= np.max(scat) # Normalize for display purposes

# Mask the direct beam
beam_mask = np.ones_like(scat)
beam_mask[detector.scattering_angles(beam=beam) < 0.01] = 0

# The simulation engine doesn't operate on "2D" arrays, so we reshape at the end:
scat = detector.reshape(scat)
beam_mask = detector.reshape(beam_mask)

# Display the diffraction intensities:
if True:
    cmap = 'CMRmap'
    fig = plt.figure(2)
    ax1 = fig.add_subplot(121)
    dispim1 = scat
    dispim1 *= beam_mask
    dispim1 = np.log(dispim1 + 1e-7)
    ax1.imshow(dispim1, cmap=plt.get_cmap(cmap))
    dispim2 = np.fft.fftshift(np.abs(np.fft.fftn(scat)))
    ax2 = fig.add_subplot(122)
    ax2.imshow(dispim2, cmap=plt.get_cmap(cmap))
    plt.show()
