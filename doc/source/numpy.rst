.. _numpy_anchor:

Working with Numpy
==================

Indexing and internal memory layout of ndarray objects
------------------------------------------------------

According to the `docs <https://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html#internal-memory-layout-of-an-ndarray>`_,
Numpy ndarray objects can accommodate any strided indexing scheme.  The bornagain package uses some
Fortran and OpenCL routines that make assumptions about the internal memory layout of ndarrays.  We assume that
all ndarrays are in the *default* "c-contiguous order", which defines a syntax in which incremements in the right-most
index of an array correspond to the smallest steps in the internal memory buffer.  The data buffer is contiguous --
there are no "gaps".  The following example illustrates this:

.. code-block:: python

    import numpy as np
    a = np.array([[1,2,3],[4,5,6]])
    print(a.shape)
    # output: (2, 3)
    print(a.flags.c_contiguous)
    # output: True
    print(a.ravel())
    # output: [1 2 3 4 5 6]
    print(a[0, 1])
    # output: 2

Arrays of vectors
-----------------

If you have *N* vectors of dimension 3, bornagain assumes they are stored with a shape of (*N*, 3).  This choice was
made because the right-most index of a numpy array has the smallest stride by default, and because it usually makes
most sense to have vector components stored close to each other in memory.  This convention is assumed in every function
in bornagain.  This note is to avoid ambiguity in the case of a (3, 3) array.

How to rotate vectors
---------------------

We must be certain that we adhere to a convention with regard to vector rotations.  If you need to rotate a vector or an
array of vectors with shape (*N*, 3) with the matrix *R*, you can do either of the following:

.. code-block:: python

    vec_rotated = np.dot(R, vec.T).T

    vec_rotated = np.dot(vec, R.T)

For clarity, here is what you should expect:

.. code-block:: python

    R = np.array([[0, 1., 0], [-1, 0, 0], [0, 0, 1.]])
    vec = np.array([1, 2, 3])
    vec_rotated = np.dot(vec, R.T)
    print(R)
    # output:
    # [[ 0.  1.  0.]
    #  [-1.  0.  0.]
    #  [ 0.  0.  1.]]
    print(vec)
    # [1 2 3]
    print(vec_rotated)
    # [ 2. -1.  3.]

Note that the above is consistent with rotation operations performed on GPU devices within the
:mod:`simulate.clcore <bornagain.simulate.clcore>` module.

Representation of 3D density/intensity maps in numpy arrays
-----------------------------------------------------------

This needs to be specified some day...