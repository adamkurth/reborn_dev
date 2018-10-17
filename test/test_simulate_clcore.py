"""
Test the clcore simulation engine in bornagain.simulate.  This requires pytest.  You can also run from main
like this: 
> python test_simulate_clcore.py
If you want to view results just add the keyword "view" 
> python test_simulate_clcore.py view
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)

import sys
import time
import numpy as np

sys.path.append('..')
try:
    from bornagain.simulate import clcore
    import pyopencl
    import bornagain as ba
    from bornagain import utils
    havecl = True
    # Check for double precision:
    core = clcore.ClCore(context=None, queue=None, group_size=1, double_precision=True)
    if core.double_precision:
        have_double = True
    else:
        have_double = False
except ImportError:
    havecl = False

view = False

if len(sys.argv) > 1:
    view = True

def test_clcore_float():

    if havecl:
        _clcore(double_precision=False)

def test_clcore_double():

    if havecl and have_double:
        _clcore(double_precision=True)

def _clcore(double_precision=False):

    ###########################################################################
    # Setup the simulation core
    ###########################################################################

    core = clcore.ClCore(context=None, queue=None, group_size=1, double_precision=double_precision)

    assert(core.get_group_size() == core.group_size)
   
    print("Using group size: %d" %core.group_size)
    ###########################################################################
    # Check that there are no errors in phase_factor_qrf_inplace
    # TODO: check that the amplitudes are correct
    ###########################################################################

    pad = ba.detector.PADGeometry()
    pad.simple_setup(n_pixels=4, pixel_size=1, distance=1)
    N = 10
    R = np.eye(3, dtype=core.real_t)
    q = pad.q_vecs(beam_vec=[0,0,1], wavelength=1)
    
    r = np.random.random([N,3])
    f = np.random.random([N])*1j

    A = core.phase_factor_qrf(q, r, f, R)
    assert(type(A) is np.ndarray)
    
    # make device arrays first
    q = core.to_device(q) 
    r = core.to_device(r)
    f = core.to_device(f, dtype=core.complex_t)
    a = core.to_device(shape=[q.shape[0]], dtype=core.complex_t)
    R = None
    
    core.phase_factor_qrf(q, r, f, R, a)
    A1 = a.get()

    for _ in range(9):
        core.phase_factor_qrf(q, r, f, R, a, add=True)
    
    A10 = a.get()
    
    assert( np.allclose(10*A1, A10))

    del q, r, f, R, N

    ###########################################################################
    # Check for errors in phase_factor_pad
    # TODO: check that amplitudes are correct
    ###########################################################################

    N = 10
    r = np.random.random([N, 3]).astype(dtype=core.real_t)
    f = np.random.random([N]).astype(dtype=core.complex_t)
    T = np.array([0, 0, 0], dtype=core.real_t)
    F = np.array([1, 0, 0], dtype=core.real_t)
    S = np.array([0, 1, 0], dtype=core.real_t)
    B = np.array([0, 0, 1], dtype=core.real_t)
    nF = 3
    nS = 4
    w = 1
    R = np.eye(3, dtype=core.real_t)

    A = core.phase_factor_pad(r, f, T, F, S, B, nF, nS, w, R, a=None)
    assert(type(A) is np.ndarray)

    r = core.to_device(r)
    f = core.to_device(f)
    a = core.to_device(shape=(nF * nS), dtype=core.complex_t)

    A = core.phase_factor_pad(r, f, T, F, S, B, nF, nS, w, R, a)

    del N, r, f, T, F, S, B, nF, nS, w, R, a, A

    ###########################################################################
    # Check for errors in phase_factor_mesh
    # TODO: check that amplitudes are correct
    ###########################################################################

    n_atoms = 10
    n_mesh = np.array([2, 3, 4])
    q_min = np.array([-1] * 3)
    q_max = np.array([+1] * 3)
    r = np.random.random([n_atoms, 3])
    f = np.random.random([n_atoms]) * 1j

    A = core.phase_factor_mesh(r, f, n_mesh, q_min, q_max)
    assert(type(A) is np.ndarray)

    r = core.to_device(r)
    f = core.to_device(f)
    a = core.to_device(shape=(np.prod(n_mesh)), dtype=core.complex_t)

    A = core.phase_factor_mesh(r, f, n_mesh, q_min, q_max, a)
    assert(type(A) is pyopencl.array.Array)

    del n_atoms, n_mesh, q_min, q_max, r, f, A, a

    ###########################################################################
    # Check for errors in buffer_mesh_lookup
    # TODO: check that amplitudes are correct
    ###########################################################################

    n_atoms = 10
    n_mesh = np.array([2, 3, 4])
    q_min = np.array([-1] * 3)
    q_max = np.array([+1] * 3)
    r = np.random.random([n_atoms, 3])
    f = np.random.random([n_atoms]) * 1j
    
    pad = ba.detector.PADGeometry()
    pad.simple_setup(n_pixels=4, pixel_size=1, distance=1)
    R = np.eye(3, dtype=core.real_t)
    q = pad.q_vecs(beam_vec=[0, 0, 1], wavelength=1)

    A = core.phase_factor_mesh(r, f, n_mesh, q_min, q_max)
    A2 = core.buffer_mesh_lookup(A, n_mesh, q_min, q_max, q)
    assert(type(A) is np.ndarray)
    assert(type(A2) is np.ndarray)

    r = core.to_device(r)
    f = core.to_device(f)
    q = core.to_device(q)
    a = core.to_device(shape=(np.prod(n_mesh)), dtype=core.complex_t)
    a_out = core.to_device(shape=pad.n_fs*pad.n_ss, dtype=core.complex_t)

    A = core.phase_factor_mesh(r, f, n_mesh, q_min, q_max, a)
    A2 = core.buffer_mesh_lookup(a, n_mesh, q_min, q_max, q, R, a_out)
    assert(type(A) is pyopencl.array.Array)
    assert(type(A2) is pyopencl.array.Array)

    del n_atoms, n_mesh, q_min, q_max, r, f, A, A2, a, q, a_out


    ###########################################################################
    # Check for errors in run_cromermann
    ###########################################################################

    # TODO: Derek, your simulators are crashing again.  Probably my fault - sorry.  -Rick

#    # simulate 1000 random numbers into 1000x1000 pixels
#    natom = n_pixels_edge = 1000
#    atom_pos = np.random.random( (natom,3) )
#    atomic_nums = np.ones(natom)
#   
##   make a simple detector 
#    #D = ba.detector.SimpleDetector(n_pixels=1000) 
#    
#    def dumb_detector(n_pixels_edge):
#        img_sh = (n_pixels_edge,n_pixels_edge)
#        py,px = np.indices(img_sh) # pixel integers
#        pr = np.sqrt( (py-n_pixels_edge/2. )**2 + (px-n_pixels_edge/2.)**2 ) # radial pixel value
#        
#        theta = np.arctan( (  pr * 0.00005/0.05 ) )/2. # 50 micron pixels, 50 mm detector distance
#        phi = np.arctan2( py-500., px-500. ) # pixel azimuthal
#        q = np.sin(theta) * 4*np.pi / 1. # 1 angstrom wavelen
#
#        qx = np.cos(theta) * q * np.cos(phi)
#        qy = np.cos(theta) * q * np.sin(phi)
#        qz = np.sin(theta) * q
#        q_vecs = np.vstack((qx.ravel(), qy.ravel(), qz.ravel())).T
#   
#        return img_sh, q_vecs
#
#    img_sh, q_vecs = dumb_detector(n_pixels_edge)
#
#    #core.prime_cromermann_simulator(D.Q, atomic_nums)
#    core.prime_cromermann_simulator(q_vecs, atomic_nums)
#    q = core.get_q_cromermann()
#    r = core.get_r_cromermann(atom_pos, sub_com=False) 
#    core.run_cromermann(q, r, rand_rot=True)
#    A = core.release_amplitudes()

def test_rotations():

    if havecl:

        theta = 25*np.pi/180.
        sin = np.sin(theta)
        cos = np.cos(theta)

        R = np.array([[cos, sin, 0],
                      [-sin, cos, 0],
                      [0, 0, 1]])

        core = clcore.ClCore(context=None, queue=None, group_size=1, double_precision=False)

        vec1 = np.array([1, 2, 0], dtype=core.real_t)
        vec2 = core.test_rotate_vec(R, vec1)
        vec3 = utils.rotate(R, vec1)

        # Rotation on gpu and rotation with utils.rotate should do the same thing
        assert(np.max(np.abs(vec2-vec3)) <= 1e-6)

        vec1 = np.array([1, 2, 0], dtype=core.real_t)
        vec2 = core.test_rotate_vec(R, vec1)
        vec4 = np.random.rand(10, 3).astype(core.real_t)
        vec4[0, :] = vec1
        vec3 = utils.rotate(R, vec4)
        vec3 = vec3[0, :]

        # Rotation on gpu and rotation with utils.rotate should do the same thing (even for many vectors; shape Nx3)
        assert(np.max(np.abs(vec2-vec3)) <= 1e-6)

        R = np.array([[0, 1.0, 0],
                      [-1.0, 0, 0],
                      [0, 0, 1.0]])

        vec1 = np.array([1.0, 0, 0], dtype=core.real_t)
        vec2 = core.test_rotate_vec(R, vec1)
        vec3 = utils.rotate(R, vec1)
        vec_pred = np.array([0, -1.0, 0])

        # Check that results are as expected
        assert(np.allclose(vec2, vec3))
        assert (np.allclose(vec2, vec_pred))


if __name__ == '__main__':

    test_clcore_float()
    test_rotations()
