r"""
GPU Simulations
===============

Some examples of how to simulate protein diffraction from a PDB file, using GPU devices.

Contributed by Richard A. Kirian.

Imports:
"""

import time
import numpy as np
from scipy import constants as const
import matplotlib.pyplot as plt
import reborn
import reborn.target.crystal as crystal
import reborn.simulate.clcore as core
from reborn.utils import rotation_about_axis

eV = const.value('electron volt')

# %%
# Our agenda here is to do the following computation of diffraction intensity:
#
# .. math::
#
#     I(\vec{q}) = J_0 \Delta \Omega r_e^2 P(\vec{q}) \sum_n f_n(q) \sum_m \exp(i \vec{q} \cdot \vec{r}_{nm})
#
# where :math:`J_0` is the incident fluence (usually photons/area or energy/area), :math:`\Delta \Omega` is the solid
# angle of the detector pixel, :math:`r_e` is the classical electron radius, :math:`P(\vec{q})` is an x-ray polarization
# correction.  These pre-factors are readily computed using the |Beam| and |PADGeometry| classes.  The
# :math:`q`-dependent scattering factors can also be generated from lookup tables as discussed in other examples.  The
# real focus of this example is the computation of the sum
#
# .. math::
#
#     I(\vec{q}) \propto \sum_m f_m(0) \exp(i \vec{q} \cdot \vec{r}_{m})
#
# which is the most time-consuming part of the calculation.
#
# We begin by setting up the OpenCL context and queue, which manage the computations on GPU devices and are bundled into
# the |ClCore| class that reborn provides.  We use the |pyopencl| package to manage GPU computations.  Note that if you
# do not have a GPU, these simulations should also work on a CPU if you install the pocl package, but obviously
# computational times will be much larger.

simcore = core.ClCore(group_size=32, double_precision=False)

# %%
# You can have a look at device parameters/options and choose the one you want.  If you are reading the auto-generated
# web documentation, this example is probably running on a *single CPU core*.  Compute times are not representative of
# compute times for GPU devices.  (BTW, If someone knows how to set up a gitlab runner with a GPU, do tell!)

core.help()

# %%
# Let's check which device we are using:
print(simcore.get_device_name())

# %%
# First we set up a pixel array detector (|PAD|) and an x-ray |Beam|:

beam = reborn.source.Beam(photon_energy=10000*eV)
pad = reborn.detector.PADGeometry(shape=(1001, 1001), pixel_size=100e-6, distance=0.5)
q_vecs = pad.q_vecs(beam=beam)
n_pixels = q_vecs.shape[0]

# %%
# Next we load a crystal structure from pdb file, from which we can get coordinates and scattering factors.

cryst = crystal.CrystalStructure('2LYZ')
r_vecs = cryst.molecule.coordinates  # These are atomic coordinates (Nx3 array)
n_atoms = r_vecs.shape[0]

# %%
# Look up atomic scattering factors (they are complex numbers).  Note that these are not :math:`q`-dependent; they are
# the forward scattering factors :math:`f(0)`.

f = cryst.molecule.get_scattering_factors(beam=beam)

# %%
# The simplest approach
# ---------------------
#
# Let's start with the simplest way of simulating diffraction from a collection of atomic coordinates.  This method
# avoids any fancy footwork with respect to managing RAM and GPU memory.  You can totally ignore the fact that the
# computation is done on a GPU.

t = time.time()
A = simcore.phase_factor_qrf(q_vecs, r_vecs, f)
print('phase_factor_qrf: %7.03f ms' % ((time.time() - t) * 1e3))

# %%
# The above takes approximately 120 ms on a laptop with a Intel(R) Gen9 HD Graphics NEO GPU (similar for most laptops).
#
# Let's have a look at the resulting pattern.  Note that the arrays that result from simulations are not 2D arrays --
# we are agnostic to the shape of the array since we simply loop over :math:`\vec{q}` vectors.

plt.imshow(np.log(pad.reshape(np.abs(A) ** 2) + 0.1), interpolation='nearest', cmap='gray', origin='lower')
plt.title('y: up, x: right, z: beam (towards you)')
plt.show()


# %%
# Pre-allocation of GPU arrays
# ----------------------------
#
# Now let's be a bit smarter -- we will first transfer our vector arrays from RAM to the GPU device global memory.
# We will also create a GPU array to store the amplitudes.
# This is a useful thing to do in the event that you plan to do more than one simulation, and you do not want to waste
# time transferring data between RAM/GPU redundantly.  Data transfer is frequently the bottleneck in GPU programs.
#
# Note: we made the choice to work in single-precision above, when we created the |ClCore| instance.  When we move
# our RAM data to the GPU, we need to make sure that the types are correct, otherwise |ClCore| will be redundantly
# transforming data types under the hood.  We use the ``real_t`` and ``complex_t`` to specify the types.

t = time.time()
q_dev = simcore.to_device(q_vecs, dtype=simcore.real_t)
r_dev = simcore.to_device(r_vecs, dtype=simcore.real_t)
f_dev = simcore.to_device(f, dtype=simcore.complex_t)
a_dev = simcore.to_device(shape=(q_dev.shape[0]), dtype=simcore.complex_t)
tf = time.time() - t
print('Move to GPU memory: %7.03f ms' % (tf * 1e3))

# %%
# The above takes about 10 ms on our test laptop.  The next line takes about 10 ms less time, as expected.

t = time.time()
simcore.phase_factor_qrf(q_dev, r_dev, f_dev, a=a_dev, add=False)
print('phase_factor_qrf: %7.03f ms' % ((time.time() - t) * 1e3))

# %%
# Previously, we accessed the amplitudes via the output of the above function.  This time, we pass in the GPU array
# directly.  In order to do further CPU operations, we need
t = time.time()
a = a_dev.get()
print("Moving amplitudes back to CPU memory in %7.03f ms" % ((time.time() - t) * 1e3))

# %%
# The above takes about 1 ms.
#
# Let's have a look at the resulting pattern, which is of course the same as the previous one:

plt.imshow(np.log(pad.reshape(np.abs(a) ** 2) + 0.1), interpolation='nearest', cmap='gray', origin='lower')
plt.title('y: up, x: right, z: beam (towards you)')
plt.show()


# %%
# Pixel-array detectors
# ---------------------
#
# We can also avoid computing :math:`\vec{q}` vectors on the CPU; they can be computed directly on the GPU, which saves
# a bit of global memory.

a_dev = simcore.to_device(shape=(n_pixels,), dtype=simcore.complex_t)

t = time.time()
simcore.phase_factor_pad(r_vecs, f, pad=pad, beam=beam, a=a_dev)
print('phase_factor_pad: %7.03f ms' % ((time.time() - t) * 1e3))

# %%
# The above takes about 130 ms on our test laptop.
#
# Let's look at the pattern, which should again be the same as the above patterns.

plt.imshow(np.log(pad.reshape(np.abs(a_dev.get()) ** 2) + 0.1), interpolation='nearest', cmap='gray', origin='lower')
plt.title('y: up, x: right, z: beam (towards you)')
plt.show()

# %%
# Summing amplitudes from multiple molecules
# ------------------------------------------
#
# Now we see how to add the amplitudes from copies of a molecule that have been rotated and translated.  First, we
# simulate the amplitudes from a molecule that has not been rotated or translated:

simcore.phase_factor_pad(r_vecs, f, pad=pad, beam=beam, a=a_dev)

# %%
# Now we add the amplitudes to a translated copy.  The phase_factor methods can do the addition on the GPU, but we need
# to use the appropriate keywords.  The ``add`` keyword indicates that the ``a_dev`` array should not be overwritten,
# but rather the simulated amplitudes should be added to the existing array.  We will first translate the molecule and
# add the previous amplitudes, which should reveal a fringe pattern running along the "x" axis.

# %%
# Here is a translation vector -- moves the atomic coordinates 5 nm in the +x direction

U = np.array([1, 0, 0])*5e-9

# %%
# Here's how we add this vector to the atomic coordinates, on the GPU, at the time of the computation:

simcore.phase_factor_pad(r_vecs, f, pad=pad, beam=beam, a=a_dev, add=True, U=U)

# %%
# Have a look at the fringe pattern.

plt.imshow(np.log(pad.reshape(np.abs(a_dev.get()) ** 2) + 0.1), interpolation='nearest', cmap='gray', origin='lower')
plt.title('y: up, x: right, z: beam (towards you)')
plt.show()

# %%
# Now let's add amplitudes from a rotated and translated copy.  We make a rotation matrix and translate along "y".

R = rotation_about_axis(20*np.pi/180.0, [1, 1, 0])
U2 = np.array([0, 1, 0])*2e-9

simcore.phase_factor_pad(r_vecs, f, pad=pad, beam=beam, a=a_dev, add=True, U=U2, R=R)

plt.imshow(np.log(pad.reshape(np.abs(a_dev.get()) ** 2) + 0.1), interpolation='nearest', cmap='gray', origin='lower')
plt.title('y: up, x: right, z: beam (towards you)')
plt.show()

# %%
# Speeding up with lookup tables
# ------------------------------
#
# If you need to do a great many simulations of the same molecule, consider doing a 3D simulation first, and then
# interpolating from that map in order to speed up your simulations.  This can reduce compute times by orders of
# magnitude.
#
# First we compute the 3D mesh of diffraction amplitudes:

q_max = np.max(pad.q_mags(beam=beam))
N = 50  # Number of samples

a_map_dev = simcore.to_device(shape=(N ** 3,), dtype=simcore.complex_t)
t = time.time()
simcore.phase_factor_mesh(r_vecs, f, N=N, q_min=-q_max, q_max=q_max, a=a_map_dev)
print('phase_factor_mesh: %7.03f ms' % ((time.time() - t) * 1e3))

# %%
# Now we slice a 2D pattern from the 3D mesh.  The slice takes about 1 ms on our test laptop, which is about 100-fold
# faster than doing the all-atom simulation.

t = time.time()
simcore.mesh_interpolation(a_map_dev, q_dev, N=N, q_min=-q_max, q_max=q_max, a=a_dev)
print('mesh_interpolation: %7.03f ms' % ((time.time() - t) * 1e3))

# %%
# Here's the result from the lookup table.  You'll notice interpolation artifacts, but it otherwise should look like
# the first pattern in this example.

plt.imshow(np.log(pad.reshape(np.abs(a_dev.get()) ** 2) + 0.1), interpolation='nearest', cmap='gray', origin='lower')
plt.title('y: up, x: right, z: beam (towards you)')
plt.show()
