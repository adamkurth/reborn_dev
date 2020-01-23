"""
Test the clcore simulation engine in bornagain.simulate.  This requires pytest.  You can also run from main
like this:
> python test_simulate_clcore.py
If you want to view results just add the keyword "view"
> python test_simulate_clcore.py view
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np
from bornagain.simulate import clcore
import pyopencl
from pyopencl import array as clarray
cl_array = clarray.Array
havecl = True
test_core = clcore.ClCore(context=None, queue=None, group_size=1, double_precision=True)
if test_core.double_precision:
    have_double = True
else:
    have_double = False


def test_clmath():

    ctx = clcore.create_some_gpu_context()

    q = pyopencl.CommandQueue(ctx)

    a = np.random.random((5, 5)).astype(np.complex64)
    a_gpu = pyopencl.array.to_device(q, a)

    b = np.random.random((5, 5)).astype(np.complex64)
    b_gpu = pyopencl.array.to_device(q, b)

    # Test addition
    c = a + b
    c_gpu = a_gpu + b_gpu
    c_gpu = c_gpu.get()
    assert np.max(np.abs(c - c_gpu)) < 1e-6

    # Test multiplication
    c = a * b
    c_gpu = a_gpu * b_gpu
    c_gpu = c_gpu.get()
    assert np.max(np.abs(c - c_gpu)) < 1e-6

    # Test exponentiation
    c = a ** 2
    c_gpu = a_gpu ** 2
    c_gpu = c_gpu.get()
    assert np.max(np.abs(c - c_gpu)) < 1e-6


# def test_atomics_01():
#
#     if not havecl:
#         return
#
#     core = clcore.ClCore(group_size=32)
#
#     n = 3
#     a = np.zeros(n)
#     b = np.arange(n)
#     a_gpu = core.to_device(a, dtype=core.real_t)
#     b_gpu = core.to_device(b, dtype=core.real_t)
#     core.test_atomic_add_real(a_gpu, b_gpu)
#     assert a_gpu.get()[0] - np.sum(b) * n == 0
#
#
# def test_atomics_02():
#
#     if not havecl:
#         return
#
#     core = clcore.ClCore(group_size=32)
#
#     n = 101
#     a = np.zeros(n)
#     b = np.arange(n)
#     a_gpu = core.to_device(a, dtype=core.int_t)
#     b_gpu = core.to_device(b, dtype=core.int_t)
#     core.test_atomic_add_int(a_gpu, b_gpu)
#     assert a_gpu.get()[0] - np.sum(b) * n == 0
#
#
# def test_atomics_03():
#
#     if not havecl:
#         return
#
#     core = clcore.ClCore(group_size=32)
#
#     n = 100
#     a = np.zeros(n)
#     b = np.arange(n)
#     a_gpu = core.to_device(a, dtype=core.real_t)
#     b_gpu = core.to_device(b, dtype=core.real_t)
#     core.test_atomic_add_real(a_gpu, b_gpu)
#     assert a_gpu.get()[0] - np.sum(b) * n == 0
#
#
# def test_atomics_04():
#
#     if not havecl:
#         return
#
#     core = clcore.ClCore(group_size=32)
#
#     n = 100
#     a = np.zeros(n)
#     b = np.arange(n)
#     a_gpu = core.to_device(a, dtype=core.real_t)
#     b_gpu = core.to_device(b, dtype=core.real_t)
#     m = 5
#     for _ in range(m):
#         core.test_atomic_add_real(a_gpu, b_gpu)
#     assert a_gpu.get()[0] - np.sum(b) * n * m == 0


def test_rotations(double_precision=False):

    # if not havecl:
    #     return

    core = clcore.ClCore(context=None, queue=None, group_size=1, double_precision=double_precision)

    theta = 25*np.pi/180.
    sin = np.sin(theta)
    cos = np.cos(theta)

    rot = np.array([[cos, sin, 0],
                    [-sin, cos, 0],
                    [0, 0, 1]], dtype=core.real_t)
    trans = np.array([1, 2, 3], dtype=core.real_t)
    vec1 = np.array([1, 2, 0], dtype=core.real_t)

    vec2 = core.test_rotate_vec(rot, trans, vec1)
    vec3 = np.dot(vec1, rot.T) + trans

    # Rotation on gpu and rotation with utils.rotate should do the same thing
    assert np.max(np.abs(vec2-vec3)) <= 1e-6

    vec1 = np.array([1, 2, 0], dtype=core.real_t)
    vec2 = core.test_rotate_vec(rot, trans, vec1)
    vec4 = np.random.rand(10, 3).astype(core.real_t)
    vec4[0, :] = vec1
    vec3 = np.dot(vec4, rot.T) + trans
    vec3 = vec3[0, :]

    # Rotation on gpu and rotation with utils.rotate should do the same thing (even for many vectors; shape Nx3)
    assert(np.max(np.abs(vec2-vec3)) <= 1e-6)

    rot = np.array([[0, 1.0, 0],
                    [-1.0, 0, 0],
                    [0, 0, 1.0]])
    trans = np.zeros((3,), dtype=core.real_t)
    vec1 = np.array([1.0, 0, 0], dtype=core.real_t)
    vec2 = core.test_rotate_vec(rot, trans, vec1)
    vec3 = np.dot(vec1, rot.T) + trans
    vec_pred = np.array([0, -1.0, 0])

    # Check that results are as expected
    assert np.max(np.abs(vec2 - vec3)) < 1e-6
    assert np.max(np.abs(vec2 - vec_pred)) < 1e-6


def test_phase_factors(double_precision=False):

    r"""
    Check that phase_factor_qrf gives the same results as phase_factor_mesh, as per documentation.

    Args:
        double_precision (bool): Check that double precision works if available
    """

    core = clcore.ClCore(context=None, queue=None, group_size=1, double_precision=double_precision)

    q_min = np.array([1, 2, 3])
    q_max = q_min + 1
    shape = np.array([2, 2, 2])
    dq = (q_max-q_min)/(shape-1)
    qx = np.arange(shape[0]) * dq[0] + q_min[0]
    qy = np.arange(shape[1]) * dq[1] + q_min[1]
    qz = np.arange(shape[2]) * dq[2] + q_min[2]
    qxx, qyy, qzz = np.meshgrid(qx, qy, qz, indexing='ij')
    q = np.vstack([qxx.ravel(), qyy.ravel(), qzz.ravel()]).T.copy()
    r = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    R = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    U = np.array([1, 2, 3])
    f = np.array([1, 2, 5])
    amps1 = core.phase_factor_mesh(r, f, q_min=q_min, q_max=q_max, N=shape, R=R, U=U)
    amps2 = core.phase_factor_qrf(q, r, f, R=R, U=U)
    assert amps1[0] == amps2[0]

    rp = np.dot(r, R.T)+U
    amps0 = 0
    for n in range(r.shape[0]):
        amps0 += f[n]*np.exp(-1j*np.dot(q[0, :], rp[n, :]))
    assert np.abs(amps0 - amps2[0])/np.abs(amps0) < 1e-4


def test_density_map():

    # import numpy as np
    from bornagain.target import crystal
    # from bornagain.simulate import clcore, atoms
    # import pyqtgraph as pg
    # from scipy import constants

    # The CrystalStructure object has a UnitCell, SpaceGroup, and other information.  The input can be any path to a PDB
    # file or it can be the name of a PDB entry.  The PDB will be fetched from the web if necessary and possible.  The
    # PDB entry 2LYZ comes with bornagain.
    cryst = crystal.CrystalStructure('2LYZ')

    # The oversampling ratio:
    osr = 2
    # The desired map resolution, which will be adjusted according to crystal lattice and sampling constraints:
    res = 2e-10
    # The CrystalDensityMap is a helper class that ensures sampling in the crystal basis is configured such that
    # the crystal spacegroup symmetry operations of a density map can be performed strictly through permutation operations.
    # Thus, no interpolations are needed for spacegroup symmetry operations.
    cdmap = crystal.CrystalDensityMap(cryst, res, osr)

    # The ClCore instance manages the GPU simulations.
    simcore = clcore.ClCore()

    # Create two atom position vectors, both at the origin.
    x_vecs = np.zeros([2, 3])
    # Now shift one of them along the "z" coordinate (in crystal basis) by n steps.  The step size comes from the
    # CrystalDensityMap, which, again, considers how to intelligently sample crystallographic density maps.
    n = 2  # np.round(1/cdmap.dx[2]).astype(int) - 1
    x_vecs[1, 2] = n * cdmap.dx[2]

    # Get some scattering factors
    f = np.array([2, 1j]) #atoms.get_scattering_factors(atomic_numbers=[6, 8], photon_energy=1e4*constants.eV)
    # f[0] = 2
    # f[1] = 1j

    ###############################################
    # METHOD 1:
    ###############################################
    # Simulate amplitudes using atomistic coordinates, structure factors, and a direct summation over
    #                              F(h) =  sum_n f_n*exp(-i 2*pi*h.x_n)
    # Recipcorcal-space coordinates are chosen such that they will correspond to a numpy FFT operation.  The limits of that
    # sample grid are provided by the CrystalDensityMap class:
    g_min = cdmap.h_min * 2 * np.pi
    g_max = cdmap.h_max * 2 * np.pi
    # Simulation tool for regular 3D grid of reciprocal-space samples.
    amps1 = simcore.phase_factor_mesh(x_vecs, f=f, q_min=g_min, q_max=g_max, N=cdmap.shape)
    # Because the phase_factor_mesh function above computes on a grid, the direct 000 voxel is centered.  We must shift
    # the array such that the h=000 is located at the first voxel as per the standard FFT arrangement in numpy.
    amps1 = np.fft.ifftshift(amps1.reshape(cdmap.shape))
    # Transforming from amplitudes to density is now a simple inverse FFT.
    dmap1 = np.fft.ifftn(amps1.astype(np.float32))

    ##################################################
    # METHOD 2:
    #################################################
    # First make the scattering density map, and then FFT the map to create amplitudes.
    dmap2 = np.zeros(cdmap.shape).astype(np.complex64)
    # Instead of defining a list of atomic coordinates, we directly set the scattering densities to the scattering factors
    # used in METHOD 1.  Note that we've chosen atomic coordinates so that they will lie exactly on grid points in our 3D
    # maps.
    dmap2[0, 0, 0] = f[0]
    dmap2[0, 0, n] = f[1]
    amps2 = np.fft.fftn(dmap2)

    def compare(a, b):
        return np.max(a - b) / np.mean((a + b) / 2)

    assert compare(np.abs(amps1), np.abs(amps2)) < 1e-5
    assert compare(np.abs(np.real(amps1)), np.abs(np.real(amps2))) < 1e-5
    assert compare(np.abs(np.imag(amps1)), np.abs(np.imag(amps2))) < 1e-5
