from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np

try:
    from bornagain.fortran import wtf_f
except ImportError:
    wtf_f = None

def test_rotations_and_broadcasting():

    rot = np.array([[0, 1., 0],
                  [-1, 0, 0],
                  [0, 0, 1.]])
    vec = np.array([1, 2, 3])
    vec2 = np.random.rand(5, 3)
    vec2[0, :] = vec

    # We are checking that the following two method of rotating vectors are the same:
    vec_rotated = np.dot(vec2, rot.T)
    vec_rotated2 = np.dot(rot, vec2.T).T

    # Also note that the result is as expected:
    assert(np.allclose(vec_rotated[0, :], np.array([2, -1, 3])))
    assert(np.allclose(vec_rotated, vec_rotated2))
    assert(np.allclose(vec_rotated[0, :], np.array([2, -1, 3])))

    # Check that addition broadcasting works even if arrays don't have shape M x D and 1 x D
    vec1 = np.array([1, 2, 3])
    vec2 = np.zeros((5, 3))
    vec3 = vec2 + vec1
    assert(np.sum(vec3[:, 1]) == 10)

    vec1 = np.array([[1, 2, 3]])
    vec2 = np.zeros((5, 3))
    vec3 = vec2 + vec1
    assert(np.sum(vec3[:, 1]) == 10)


def test_fortranarray():

    c = np.arange(200).reshape(10, 20).copy()
    f = np.asfortranarray(c)
    c_original = c               # We need to keep track of the original arrays
    f_original = f
    c_flat = c.reshape(200)
    f_flat = f.reshape(200)

    assert c.data != c_flat.data
    assert f.data != f_flat.data
    assert c.data == f.data      # ndarray.data is: "Python buffer object pointing to the start of the array’s data."
    assert c_flat.data == f_flat.data
    assert c.shape[0] == 10      # They are still the same shape
    assert f.shape[0] == 10
    assert c.flags.c_contiguous  # Yet one is "c-contiguous" while the other is "f-contiguous"
    assert f.flags.f_contiguous  # This seems impossible to me.  How can c and f have the same shape and same data
                                 # without both of them being either c *or* f contiguous?
    assert c_flat[21] == 21      # Ravel makes the ND array into a 1D array, may or may not copy memory buffer.
    assert f_flat[21] == 21
    assert c_flat.data == f_flat.data
    assert c.data != c_flat.data
    assert f.data != f_flat.data
    assert c[0, 1] == 1
    c[0, 1] = 0                  # Now we modify the original array c
    assert c.flags.c_contiguous  # Still f/c contiguous as before
    assert f.flags.f_contiguous
    assert c.data != f.data      # But they no longer share the same memory.  The operation on c created a new array.
    assert c.data == c_original.data
    assert f.data == f_original.data


def test_fortran_wtf():

    # This is important to note when using f2py:
    #
    # "In general, if a NumPy array is proper-contiguous and has a proper type then it is directly passed to wrapped
    # Fortran/C function. Otherwise, an element-wise copy of an input array is made and the copy, being
    # proper-contiguous and with proper type, is used as an array argument."
    #
    # The tests below show how the above can cause a lot of confusion...

    if wtf_f is None:
        return

    # The fortran function "wtf" does the following:
    # out1(2) = 10
    # out2(2,1) = 10
    # out3(2,1,1) = 10

    out1 = np.zeros(10)
    out2 = np.zeros((10, 10))
    out3 = np.zeros((10, 10, 10))
    wtf_f.wtf(np.asfortranarray(out1), np.asfortranarray(out2), np.asfortranarray(out3))
    assert out1.flags.f_contiguous
    assert out1[1] == 10  # A 1D array can be passed into a fortran function and the function can modify the data
    assert out2.flags.c_contiguous
    assert out2[1, 0] != 10  # Look: the asfortranarray function will not let you modify the data; it makes a copy
    assert out3.flags.c_contiguous
    assert out3[1, 0, 0] != 10  # Once again, a copy is made.  Note that this issue pertains to multi-dimensional arrays

    out1 = np.zeros(10)
    out2_0 = np.zeros((10, 10))
    out2 = np.asfortranarray(out2_0)
    assert out2_0.data == out2.data  # This line shows that asfortranarray does not make a data copy immediately
    out3 = np.zeros((10, 10, 10))
    wtf_f.wtf(np.asfortranarray(out1), out2, np.asfortranarray(out3))
    assert out1.flags.f_contiguous
    assert out1[1] == 10
    assert out2.flags.f_contiguous
    assert out2[1, 0] == 10
    assert out2_0.data != out2.data  # Compare to the above - the wtf function
    assert not out3.flags.f_contiguous
    assert out3[1, 0, 0] != 10

    out1 = np.zeros(10)
    out2 = np.asfortranarray(np.zeros((10, 10)))
    out3 = np.zeros((10, 10, 10)).T  # This array is now fortran contiguous, as a result of the transpose.
    wtf_f.wtf(out1, out2, out3)
    assert out1.flags.f_contiguous
    assert out1[1] == 10
    assert out2.flags.f_contiguous
    assert out2[1, 0] == 10
    assert out3.flags.f_contiguous
    assert out3[1, 0, 0] == 10

    # Here is the proper way to do it:
    out1 = np.zeros(10)  # Make normal numpy arrays.  No "asfortrancontiguous" confusion...
    out2 = np.zeros((10, 10))
    out3 = np.zeros((10, 10, 10))
    assert out1.flags.c_contiguous  # Check for normal numpy arrays.
    assert out2.flags.c_contiguous
    assert out3.flags.c_contiguous
    wtf_f.wtf(out1.T, out2.T, out3.T)  # Pass in the transposes.
    assert out1[1] == 10  # THINK about what the fortran code does: modifies first element in memory.  Check this.
    assert out2[0, 1] == 10
    assert out3[0, 0, 1] == 10
