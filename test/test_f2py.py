import numpy as np
from bornagain.fortran import wtf_f


def test_fortran_wtf():

    # This is important to note when using f2py:
    #
    # "In general, if a NumPy array is proper-contiguous and has a proper type then it is directly passed to wrapped
    # Fortran/C function. Otherwise, an element-wise copy of an input array is made and the copy, being
    # proper-contiguous and with proper type, is used as an array argument."
    #
    # The tests below show how the above can cause a lot of confusion...

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
