"""
Test the clcore simulation engine in bornagain.simulate.  This requires pytest.  You can also run from main
like this: 
> python test_simulate_clcore.py
If you want to view results just add the keyword "view" 
> python test_simulate_clcore.py view
"""


import sys

import numpy as np
import pytest

sys.path.append('..')
try:
    from bornagain.simulate import clcore
    import pyopencl
    import bornagain as ba
    havecl = True
    # Check for double precision:
    core = clcore.ClCore(context=None, queue=None, group_size=1,
                         double_precision=True)
    if core.double_precision:
        have_double = True
    else:
        have_double = False
except ImportError:
    havecl = False

view = False

if len(sys.argv) > 1:
    view = True
    import matplotlib.pyplot as plt


clskip = pytest.mark.skipif(havecl is False, reason="Requires pyopencl module")



# @clskip
# def test_two_atoms(main=False):
#
#     """
#     Test if the simulated diffraction between two points is sensible.
#     """
#
#     r = np.zeros([2,3])
#     r[1,0] = 1e-10
#
#     f = np.ones([2])
#
#     R = np.eye(3)
#
#     q = np.zeros([2,3])
#
#     A = clcore.phase_factor_qrf(q, r, f, R)
#
#     I = np.abs(A)**2
#
#     assert(np.max(I) == 4.0)

# @clskip
# def test_equivalence_pad_qrf(main=False):
#
#     """
#     Test if we get the same results using phase_factor_qrf and phase_factor_pad
#     """
#
#     nF = np.int(50)
#     nS = np.int(101)
#     pix = np.float32(100e-6)
#     w = np.float32(1.5e-10)
#     F = np.array([1,0,0],dtype=np.float32)*pix
#     S = np.array([0,1,0],dtype=np.float32)*pix
#     B = np.array([0,0,1],dtype=np.float32)
#     T = np.array([0,0,0.1],dtype=np.float32)
#     R = np.eye(3,dtype=np.float32)
#
#     p = ba.detector.Panel()
#     p.nF = nF
#     p.nS = nS
#     p.F = F
#     p.S = S
#     p.beam.B = B
#     p.beam.wavelength = w
#     p.T = T
#
#     # Scattering vectors generated from panel object
#     q = np.float32(p.Q)
#
#     # Atomic coordinates
#     N = 5
#     r = np.zeros([N,3],dtype=np.float32)
#     r[1,0] = 1000e-10
#     r[2,1] = 1000e-10
#
#     # Scattering factors
#     f = np.ones([N],dtype=np.float32)
#
#     # Compute diffraction amplitudes
#     A1 = clcore.phase_factor_qrf(q,r,f)
#     A2 = clcore.phase_factor_pad(r,f,T,F,S,B,nF,nS,w,R)
#     A1 = A1.astype(np.complex64)
#     A2 = A2.astype(np.complex64)
#     dif = np.max(np.abs(A1-A2))
#
#     if main:
#         print('test_equivalence_pad_qrf max error: %g' % (dif))
#         if view:
#             print('Showing qrf and pad')
#             A1 = np.reshape(np.abs(A1)**2,[nS,nF])
#             A2 = np.reshape(np.abs(A2)**2,[nS,nF])
#             plt.imshow(A1,cmap='gray',interpolation='nearest')
#             plt.show()
#             plt.imshow(A2,cmap='gray',interpolation='nearest')
#             plt.show()
#             plt.imshow(A1-A2,cmap='gray',interpolation='nearest')
#             plt.show()
#
#     # This is a pretty weak tolerance... something is probably wrong
#     assert(dif <= 1e-4)

# @clskip
# def test_rotations(main=False):
#
#     """
#     Test that rotations go in the correct direction.
#     """
#
#     R = ba.utils.random_rotation_matrix()
#
#     pl = ba.detector.PanelList()
#     pl.simple_setup(100,101,100e-6,0.1,1.5e-9)
#     q = pl[0].Q
#
#     r = np.random.random([5,3])*1e-10
#     f = np.random.random([5])
#
#     q = q.astype(np.float32)
#     R = R.astype(np.float32)
#     qR = q.dot(R.T)
#
#     A1 = clcore.phase_factor_qrf(q,r,f,R)
#     A2 = clcore.phase_factor_qrf(qR,r,f)
#
#     dif = np.max(np.abs(A1-A2))
#
#     if main:
#         print("test_rotations: max error is %g" % (dif))
#
#     assert(dif < 1e-5)

def run_test_ClCore(main=False,double_precision=False):

    ###########################################################################
    # Check that we can set the group size and it propagates to the device
    ###########################################################################

    core = clcore.ClCore(context=None, queue=None, group_size=1)
    assert(core.get_group_size() == core.group_size)
    
    ###########################################################################
    # Setup the simulation core
    ###########################################################################

    core = clcore.ClCore(context=None, queue=None, group_size=1,
                         double_precision=double_precision)

    ###########################################################################
    # Check that there are no errors in phase_factor_qrf_inplace
    # TODO: actually check that the amplitudes are correct
    ###########################################################################

    pl = ba.detector.PanelList()
    pl.simple_setup(nF=3, nS=4, pixel_size=1, distance=1, wavelength=1)
    N = 10
    R = np.eye(3, dtype=core.real_t)
    q = pl[0].Q
    
    r = np.random.random([N,3])
    f = np.random.random([N])*1j
    
    core.init_amps(Npix=pl.n_pixels)
    core.phase_factor_qrf_inplace(q,r,f,R)
    assert(type(core.a_dev) is pyopencl.array.Array)
    A = core.release_amps(reset=True)
    assert(type(A) is np.ndarray)
    
#   make device arrays first
    q = core.to_device(q) 
    r = core.to_device(r)
    f = core.to_device(f)
    
    core.phase_factor_qrf_inplace(q,r,f,R)
    A1 = core.release_amps(reset=False)
    
    core.phase_factor_qrf_inplace(q,r,f,R)
    A2 = core.release_amps(reset=False)
    
    assert( np.allclose(2*A1,A2))

    core.init_amps(pl.n_pixels)
    for _ in xrange(10):
        core.phase_factor_qrf_inplace(q,r,f,R)
    
    A10 = core.release_amps()
    
    assert( np.allclose(10*A1,A10))

    del q, r, f, R, N, pl
    
    ###########################################################################
    # Check for errors in phase_factor_pad
    # TODO: check that amplitudes are correct
    ###########################################################################

    N = 10
    R = np.eye(3, dtype=core.real_t)
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
    
    pl = ba.detector.PanelList()
    pl.simple_setup(nF=3, nS=4, pixel_size=1, distance=1, wavelength=1)
    R = np.eye(3, dtype=core.real_t)
    q = pl[0].Q

    A = core.phase_factor_mesh(r, f, n_mesh, q_min, q_max)
    A2 = core.buffer_mesh_lookup(A, n_mesh, q_min, q_max, q)
    assert(type(A) is np.ndarray)
    assert(type(A2) is np.ndarray)


    r = core.to_device(r)
    f = core.to_device(f)
    q = core.to_device(q)
    a = core.to_device(shape=(np.prod(n_mesh)), dtype=core.complex_t)
    a_out = core.to_device(shape=(pl.n_pixels), dtype=core.complex_t)

    A = core.phase_factor_mesh(r, f, n_mesh, q_min, q_max, a)
    A2 = core.buffer_mesh_lookup(a, n_mesh, q_min, q_max, q, R, a_out)
    assert(type(A) is pyopencl.array.Array)
    assert(type(A2) is pyopencl.array.Array)

    del n_atoms, n_mesh, q_min, q_max, r, f, A, A2, a, q, a_out


    ###########################################################################
    # Check for errors in run_cromermann
    ###########################################################################

    # simulate 1000 random numbers into 1000x1000 pixels
    natom = n_pixels_edge = 1000
    atom_pos = np.random.random( (natom,3) )
    atomic_nums = np.ones(natom)
   
#   make a simple detector 
    #D = ba.detector.SimpleDetector(n_pixels=1000) 
    
    def dumb_detector(n_pixels_edge):
        img_sh = (n_pixels_edge,n_pixels_edge)
        py,px = np.indices(img_sh) # pixel integers
        pr = np.sqrt( (py-n_pixels_edge/2. )**2 + (px-n_pixels_edge/2.)**2 ) # radial pixel value
        
        theta = np.arctan( (  pr * 0.00005/0.05 ) )/2. # 50 micron pixels, 50 mm detector distance
        phi = np.arctan2( py-500., px-500. ) # pixel azimuthal
        q = np.sin(theta) * 4*np.pi / 1. # 1 angstrom wavelen

        qx = np.cos(theta) * q * np.cos(phi)
        qy = np.cos(theta) * q * np.sin(phi)
        qz = np.sin(theta) * q
        q_vecs = np.vstack((qx.ravel(), qy.ravel(), qz.ravel())).T
   
        return img_sh, q_vecs

    img_sh, q_vecs = dumb_detector(n_pixels_edge)

    #core.prime_cromermann_simulator(D.Q, atomic_nums)
    core.prime_cromermann_simulator(q_vecs, atomic_nums)
    q = core.get_q_cromermann()
    r = core.get_r_cromermann(atom_pos, sub_com=False) 
    core.run_cromermann(q, r, rand_rot=True)
    A = core.release_amplitudes()


@clskip
def test_ClCore(main=False):
    
    # Frist test single precision
    run_test_ClCore(main,double_precision=False)
    
    # Now test double precision if available
    if have_double:
        run_test_ClCore(main,double_precision=True)


if __name__ == '__main__':

    print('Running as main')
    main = True
#     test_two_atoms(main)
#     test_equivalence_pad_qrf(main)
#     test_rotations(main)
    test_ClCore(main)

