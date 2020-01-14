.. _working_with_fortran:

Fortran
=======

Integration of Fortran and Numpy
--------------------------------

The f2py utility included with numpy makes it quite easy to integrate simple Fortran code with Numpy.  Typically,
we wish to pass memory buffers from numpy ndarrays into a Fortran subroutine, and we modify those buffers with Fortran.
There are some very annoying issues that can arise because the ways in which the Numpy package manipulates
the inernal memory buffers of ndarrays, which might surprise you.  These under-the-hood manipulations might be
harmless... until the day you really care about operating directly on memory buffers. Examples of such complications can
be found in the `test_fortran.py` unit test.

Another matter is the way that numpy arrays are passed to fortran routines when you use f2py.  The
`documentation <https://www.numpy.org/devdocs/f2py/python-usage.html>`_ states the following:

    "*In general, if a NumPy array is proper-contiguous and has a proper type then it is directly passed to wrapped
    Fortran/C function. Otherwise, an element-wise copy of an input array is made and the copy, being proper-contiguous
    and with proper type, is used as an array argument.*"

Given the above we've come up with the following recipe to avoid possible issues:

(1) Always work with the default C-contiguous ndarray memory layout in Python code.

(2) Use assert statements in function wrappers: e.g. assert a.flags.c_contiguous == True.

(3) Transpose ndarrays before passing them to Fortran routines.  This will *not* copy memory.

(4) In your Fortran code, simply reverse the ordering of your indices as compared to your Numpy code.

Although it may be inconvenient to reverse your indexing when going between the Fortran and Python code, bear in mind
that this can only be avoided by (a) making copies of array memory, or (b) enforcing a consistent non-default internal
memory layout for all Numpy arrays that touch a Fortran routine.  Both options (a) and (b) are highly undesirable.  We
choose option (c), reverse the index order, because it holds the big advantage that we get to think about memory in the
most natural way for both Numpy *and* Fortran coding, rather than insisting that Fortran and Numpy syntax *look* the
same at the expense of speed and potential memory issues.